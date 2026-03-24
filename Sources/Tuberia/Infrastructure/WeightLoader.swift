@preconcurrency import MLX
import SwiftAcervo
import Foundation

/// Loads safetensors files into `ModuleParameters` with key remapping, tensor transforms,
/// and quantization. The single centralized loading path -- no pipe segment ever parses
/// safetensors or accesses files directly.
///
/// Loading pipeline (internal to WeightLoader):
/// 1. Access the component's directory via Acervo
/// 2. Find all `.safetensors` files (handles sharded weights)
/// 3. For each safetensors file, for each key:
///    a. `keyMapping(originalKey)` -> remappedKey (nil = skip)
///    b. `tensorTransform?(remappedKey, tensor)` -> transformed tensor (nil = identity)
///    c. Apply quantization per `QuantizationConfig`
/// 4. Collect all key-tensor pairs -> `ModuleParameters`
public struct WeightLoader: Sendable {

    /// Load weights from an Acervo component's safetensors files.
    ///
    /// The component is identified by its Acervo model ID. File access is scoped
    /// through `AcervoManager.shared.withModelAccess` -- URLs are never stored beyond
    /// the closure scope.
    ///
    /// - Parameters:
    ///   - componentId: The Acervo component/model ID to load from.
    ///   - keyMapping: Closure mapping safetensors keys to module keys. Return `nil` to skip a key.
    ///   - tensorTransform: Optional per-tensor transform applied after key remapping, before quantization.
    ///   - quantization: Quantization strategy to apply to loaded tensors.
    /// - Returns: Remapped, optionally quantized parameter tensors ready for module assignment.
    /// - Throws: `PipelineError.modelNotDownloaded` if the component is not ready in Acervo.
    ///           `PipelineError.weightLoadingFailed` on parse or I/O errors.
    public static func load(
        componentId: String,
        keyMapping: KeyMapping,
        tensorTransform: TensorTransform? = nil,
        quantization: QuantizationConfig = .asStored
    ) async throws -> ModuleParameters {
        do {
            let result = try await AcervoManager.shared.withModelAccess(componentId) {
                directoryURL -> ModuleParameters in
                let safetensorsURLs = findSafetensorsFiles(in: directoryURL)

                guard !safetensorsURLs.isEmpty else {
                    throw PipelineError.weightLoadingFailed(
                        component: componentId,
                        reason: "No .safetensors files found in component directory"
                    )
                }

                var allParameters: [String: MLXArray] = [:]

                for url in safetensorsURLs {
                    let rawArrays = try loadArrays(url: url)

                    for (originalKey, tensor) in rawArrays {
                        // Step 1: Key remapping (nil = skip)
                        guard let remappedKey = keyMapping(originalKey) else {
                            continue
                        }

                        // Step 2: Tensor transform (if provided)
                        var transformed = tensor
                        if let transform = tensorTransform {
                            transformed = transform(remappedKey, transformed)
                        }

                        // Step 3: Quantization
                        transformed = applyQuantization(transformed, config: quantization)

                        allParameters[remappedKey] = transformed
                    }
                }

                return ModuleParameters(parameters: allParameters)
            }

            return result

        } catch let error as PipelineError {
            throw error
        } catch {
            // Map Acervo errors to PipelineError
            let errorString = String(describing: error)
            if errorString.contains("NotDownloaded") || errorString.contains("notDownloaded")
                || errorString.contains("notRegistered") || errorString.contains("invalidModelId")
            {
                throw PipelineError.modelNotDownloaded(component: componentId)
            }
            throw PipelineError.weightLoadingFailed(
                component: componentId,
                reason: errorString
            )
        }
    }

    /// Load weights from a local file path (used for LoRA adapters not registered in Acervo).
    ///
    /// - Parameters:
    ///   - path: Local filesystem path to a safetensors file or directory containing safetensors files.
    ///   - keyMapping: Closure mapping safetensors keys to module keys. Return `nil` to skip a key.
    ///   - tensorTransform: Optional per-tensor transform.
    ///   - quantization: Quantization strategy.
    /// - Returns: Remapped parameter tensors.
    /// - Throws: `PipelineError.weightLoadingFailed` on parse or I/O errors.
    public static func loadFromPath(
        _ path: String,
        keyMapping: KeyMapping,
        tensorTransform: TensorTransform? = nil,
        quantization: QuantizationConfig = .asStored
    ) throws -> ModuleParameters {
        let url = URL(fileURLWithPath: path)
        var urls: [URL]

        var isDir: ObjCBool = false
        if FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue {
            urls = findSafetensorsFiles(in: url)
        } else {
            urls = [url]
        }

        guard !urls.isEmpty else {
            throw PipelineError.weightLoadingFailed(
                component: path,
                reason: "No .safetensors files found at path"
            )
        }

        do {
            var allParameters: [String: MLXArray] = [:]

            for fileURL in urls {
                let rawArrays = try loadArrays(url: fileURL)

                for (originalKey, tensor) in rawArrays {
                    guard let remappedKey = keyMapping(originalKey) else {
                        continue
                    }

                    var transformed = tensor
                    if let transform = tensorTransform {
                        transformed = transform(remappedKey, transformed)
                    }

                    transformed = applyQuantization(transformed, config: quantization)
                    allParameters[remappedKey] = transformed
                }
            }

            return ModuleParameters(parameters: allParameters)
        } catch let error as PipelineError {
            throw error
        } catch {
            throw PipelineError.weightLoadingFailed(
                component: path,
                reason: String(describing: error)
            )
        }
    }

    // MARK: - Private Helpers

    /// Find all `.safetensors` files in a directory.
    private static func findSafetensorsFiles(in directory: URL) -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
        ) else {
            return []
        }

        var urls: [URL] = []
        for case let fileURL as URL in enumerator {
            if fileURL.pathExtension == "safetensors" {
                urls.append(fileURL)
            }
        }

        return urls.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Apply quantization to a tensor according to the configuration.
    private static func applyQuantization(
        _ tensor: MLXArray, config: QuantizationConfig
    ) -> MLXArray {
        switch config {
        case .asStored:
            return tensor
        case .float16:
            return tensor.asType(.float16)
        case .bfloat16:
            return tensor.asType(.bfloat16)
        case .int4(let groupSize):
            return quantizeIfEligible(tensor, bits: 4, groupSize: groupSize)
        case .int8(let groupSize):
            return quantizeIfEligible(tensor, bits: 8, groupSize: groupSize)
        }
    }

    /// Only quantize 2D tensors with dimensions that are multiples of 32.
    /// Non-eligible tensors (biases, layer norms, embeddings) are returned as-is.
    private static func quantizeIfEligible(
        _ tensor: MLXArray, bits: Int, groupSize: Int
    ) -> MLXArray {
        // MLX quantize() only supports 2D inputs with dimensions that are multiples of 32
        guard tensor.ndim == 2 else {
            return tensor
        }

        let rows = tensor.dim(0)
        let cols = tensor.dim(1)
        guard rows % 32 == 0 && cols % 32 == 0 else {
            return tensor
        }

        // Quantize and immediately dequantize to get the quantized representation
        // in a float format that can be used with standard operations.
        // The actual quantized storage is handled by MLX's QuantizedLinear layers
        // at the model level. Here we perform dtype conversion as a proxy.
        let (wq, scales, biases) = quantized(tensor, groupSize: groupSize, bits: bits)
        return dequantized(wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
    }
}
