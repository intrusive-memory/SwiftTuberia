import Foundation
@preconcurrency import MLX
import SwiftAcervo

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
        print("[WeightLoader] withModelAccess directory for '\(componentId)': \(directoryURL.path)")
        let safetensorsURLs = findSafetensorsFiles(in: directoryURL)
        print("[WeightLoader] findSafetensorsFiles returned \(safetensorsURLs.count) file(s)")

        guard !safetensorsURLs.isEmpty else {
          throw PipelineError.weightLoadingFailed(
            component: componentId,
            reason: "No .safetensors files found in component directory"
          )
        }

        // If in a macOS App Group Container, MACF blocks fopen() for unentitled processes
        // (e.g. xctest without com.apple.security.application-groups). The guard is:
        //
        //   1. The directory is inside a Group Containers path (App Group container).
        //   2. VINETAS_TEST_MODELS_DIR is explicitly set (only the Makefile GPU/fixture
        //      test targets set this — unit tests never do, so they stay isolated from
        //      /tmp hardlinks).
        //
        // NOTE: We intentionally do NOT check `canEnumerateDirectory` here.
        // macOS MACF blocks fopen() (file content reads) but still allows opendir/stat,
        // so `FileManager.enumerator` returns results even when `fopen()` would fail.
        // Using !canEnumerateDirectory as the bypass signal is wrong — it would prevent
        // the bypass from firing on machines where the directory CAN be enumerated but
        // fopen() is still blocked by MACF (which is the common case on developer Macs).
        // The VINETAS_TEST_MODELS_DIR guard is sufficient: only GPU test targets set it.
        let effectiveURLs: [URL]
        if directoryURL.path.contains("/Group Containers/"),
          let baseDir = ProcessInfo.processInfo.environment["VINETAS_TEST_MODELS_DIR"]
        {
          let tempDir = URL(fileURLWithPath: baseDir).appendingPathComponent(componentId)
          let tempURLs = findSafetensorsFiles(in: tempDir)
          if !tempURLs.isEmpty {
            print("[WeightLoader] Using pre-hardlinked files from \(tempDir.path)")
            effectiveURLs = tempURLs
          } else {
            print("[WeightLoader] No pre-hardlinked files in \(tempDir.path), using original")
            effectiveURLs = safetensorsURLs
          }
        } else {
          effectiveURLs = safetensorsURLs
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
  /// File access is scoped through `AcervoManager.shared.withLocalAccess` — no direct
  /// `FileManager` or path construction occurs in Tuberia.
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
    } catch {
      throw PipelineError.weightLoadingFailed(
        component: path,
        reason: String(describing: error)
      )
    }
  }

  // MARK: - Private Helpers

  /// Returns true if the calling process can enumerate (opendir) the given directory.
  ///
  /// Used to distinguish between an entitled process that has App Group container access
  /// (returns `true`) and an unentitled xctest process that can stat but not open files
  /// in the container (returns `false`).
  private static func canEnumerateDirectory(_ directory: URL) -> Bool {
    guard
      let enumerator = FileManager.default.enumerator(
        at: directory,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
      )
    else { return false }
    return enumerator.nextObject() != nil
  }

  /// Find all `.safetensors` files in a directory.
  ///
  /// Tries `FileManager.enumerator` first (fast, works when the process can opendir the
  /// directory). Falls back to a stat-based probe when opendir is denied — e.g. when
  /// the directory lives inside a macOS App Group container and the test process lacks
  /// the `com.apple.security.application-groups` entitlement. The probe covers the two
  /// HuggingFace naming conventions: a single `model.safetensors` file and sharded files
  /// named `model-NNNNN-of-MMMMM.safetensors`.
  private static func findSafetensorsFiles(in directory: URL) -> [URL] {
    let fm = FileManager.default

    // Fast path: enumerate with opendir (works in non-sandboxed / entitled contexts).
    if let enumerator = fm.enumerator(
      at: directory,
      includingPropertiesForKeys: nil,
      options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants]
    ) {
      var urls: [URL] = []
      for case let fileURL as URL in enumerator {
        if fileURL.pathExtension == "safetensors" {
          urls.append(fileURL)
        }
      }
      if !urls.isEmpty {
        return urls.sorted { $0.lastPathComponent < $1.lastPathComponent }
      }
      // Enumerator returned 0 results — may be due to Container Manager blocking opendir
      // on macOS App Group containers when the process lacks the entitlement.
      // Fall through to stat-based probe to check if files actually exist.
    }

    // Fallback: stat-based probe for HuggingFace naming conventions.
    // This works even when opendir is denied (e.g. macOS App Group container
    // accessed from an unentitled test process).
    print("[WeightLoader] stat-based probe for: \(directory.path)")

    // Pattern 1: single-shard model.safetensors
    let singleFile = directory.appendingPathComponent("model.safetensors")
    let singleExists = fm.fileExists(atPath: singleFile.path)
    print("[WeightLoader] model.safetensors exists: \(singleExists)")
    if singleExists {
      return [singleFile]
    }

    // Pattern 2: sharded model-NNNNN-of-MMMMM.safetensors — probe total shard count 1…99
    for totalShards in 1...99 {
      let firstName = String(format: "model-00000-of-%05d.safetensors", totalShards)
      let firstURL = directory.appendingPathComponent(firstName)
      guard fm.fileExists(atPath: firstURL.path) else { continue }

      // Found the shard count. Build the full shard list.
      var shards: [URL] = []
      for i in 0..<totalShards {
        let name = String(format: "model-%05d-of-%05d.safetensors", i, totalShards)
        shards.append(directory.appendingPathComponent(name))
      }
      return shards
    }

    return []
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
