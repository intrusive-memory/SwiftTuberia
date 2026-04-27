import Foundation
@preconcurrency import MLX
import SwiftAcervo

/// Loads safetensors files into `ModuleParameters` with key remapping, tensor transforms,
/// and quantization. The single centralized loading path -- no pipe segment ever parses
/// safetensors or accesses files directly.
///
/// Loading pipeline (internal to WeightLoader):
/// 1. Access the component's directory via Acervo v2 `withComponentAccess`
/// 2. Find all `.safetensors` files (handles sharded weights)
/// 3. For each safetensors file, for each key:
///    a. `keyMapping(originalKey)` -> remappedKey (nil = skip)
///    b. `tensorTransform?(remappedKey, tensor)` -> transformed tensor (nil = identity)
///    c. Apply quantization per `QuantizationConfig`
/// 4. Collect all key-tensor pairs -> `ModuleParameters`
public struct WeightLoader: Sendable {

  /// Load weights from an Acervo component's safetensors files.
  ///
  /// The component is identified by its Acervo component ID. File access is scoped
  /// through `AcervoManager.shared.withComponentAccess` -- URLs are never stored beyond
  /// the closure scope. This ensures integrity verification is performed.
  ///
  /// - Parameters:
  ///   - componentId: The Acervo component ID to load from.
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
      let result = try await AcervoManager.shared.withComponentAccess(componentId) {
        handle -> ModuleParameters in
        print("[WeightLoader] withComponentAccess for component '\(componentId)'")

        // Resolve safetensors URLs from the component handle via v2 API.
        // The handle abstracts away the storage location and applies integrity checks.
        let effectiveURLs: [URL]
        do {
          effectiveURLs = try handle.urls(matching: ".safetensors")
        } catch {
          // No safetensors files found in descriptor
          throw PipelineError.weightLoadingFailed(
            component: componentId,
            reason: "Component descriptor has no .safetensors files: \(error)"
          )
        }

        print("[WeightLoader] effective safetensors count: \(effectiveURLs.count)")

        guard !effectiveURLs.isEmpty else {
          throw PipelineError.weightLoadingFailed(
            component: componentId,
            reason: "No .safetensors files found in component directory"
          )
        }

        var allParameters: [String: MLXArray] = [:]

        for url in effectiveURLs {
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
    } catch AcervoError.componentNotDownloaded(let id) {
      throw PipelineError.modelNotDownloaded(component: id)
    } catch AcervoError.componentNotRegistered(let id) {
      throw PipelineError.modelNotDownloaded(component: id)
    } catch AcervoError.componentNotHydrated(let id) {
      throw PipelineError.modelNotDownloaded(component: id)
    } catch AcervoError.integrityCheckFailed(let file, let expected, let actual) {
      throw PipelineError.weightLoadingFailed(
        component: componentId,
        reason: "Integrity check failed for '\(file)': expected SHA-256 '\(expected)', got '\(actual)'"
      )
    } catch {
      throw PipelineError.weightLoadingFailed(
        component: componentId,
        reason: String(describing: error)
      )
    }
  }

  /// Load weights from a local file path (used for LoRA adapters not registered in Acervo).
  ///
  /// File access is scoped through `AcervoManager.shared.withLocalAccess` — no direct
  /// file system access or path construction occurs in Tuberia.
  ///
  /// - Parameters:
  ///   - path: Local filesystem path to a safetensors file or directory containing safetensors files.
  ///   - keyMapping: Closure mapping safetensors keys to module keys. Return `nil` to skip a key.
  ///   - tensorTransform: Optional per-tensor transform.
  ///   - quantization: Quantization strategy.
  /// - Returns: Remapped parameter tensors.
  /// - Throws: `PipelineError.weightLoadingFailed` on I/O or parse errors.
  public static func loadFromPath(
    _ path: String,
    keyMapping: KeyMapping,
    tensorTransform: TensorTransform? = nil,
    quantization: QuantizationConfig = .asStored
  ) async throws -> ModuleParameters {
    let url = URL(fileURLWithPath: path)
    do {
      return try await AcervoManager.shared.withLocalAccess(url) { handle in
        let fileURLs = try handle.urls(matching: ".safetensors")
        guard !fileURLs.isEmpty else {
          throw PipelineError.weightLoadingFailed(
            component: path,
            reason: "No .safetensors files found at path"
          )
        }
        var allParameters: [String: MLXArray] = [:]
        for fileURL in fileURLs {
          for (originalKey, tensor) in try loadArrays(url: fileURL) {
            guard let remappedKey = keyMapping(originalKey) else { continue }
            var transformed = tensor
            if let transform = tensorTransform { transformed = transform(remappedKey, transformed) }
            allParameters[remappedKey] = applyQuantization(transformed, config: quantization)
          }
        }
        return ModuleParameters(parameters: allParameters)
      }
    } catch let error as PipelineError {
      throw error
    } catch AcervoError.componentNotDownloaded(let id) {
      throw PipelineError.modelNotDownloaded(component: id)
    } catch AcervoError.componentNotRegistered(let id) {
      throw PipelineError.modelNotDownloaded(component: id)
    } catch AcervoError.componentNotHydrated(let id) {
      throw PipelineError.modelNotDownloaded(component: id)
    } catch AcervoError.integrityCheckFailed(let file, let expected, let actual) {
      throw PipelineError.weightLoadingFailed(
        component: path,
        reason: "Integrity check failed for '\(file)': expected SHA-256 '\(expected)', got '\(actual)'"
      )
    } catch {
      throw PipelineError.weightLoadingFailed(
        component: path,
        reason: String(describing: error)
      )
    }
  }

  // MARK: - Private Helpers

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
