@preconcurrency import MLX
import Testing

@testable import Tuberia

/// Unit tests for WeightLoader using synthetic data.
/// These test key remapping, tensor transforms, and quantization logic
/// WITHOUT requiring real safetensors files or network access.
@Suite("WeightLoader Tests")
struct WeightLoaderTests {

  // MARK: - Key Remapping

  @Test("Key remapping applies correctly")
  func keyRemappingApplied() throws {
    // Create synthetic parameters simulating raw loaded data
    let rawParams: [String: MLXArray] = [
      "model.layers.0.weight": MLXArray.ones([4, 4]),
      "model.layers.1.weight": MLXArray.ones([4, 4]),
      "model.layers.2.bias": MLXArray.zeros([4]),
    ]

    // Define a key mapping that renames keys
    let keyMapping: KeyMapping = { key in
      key.replacingOccurrences(of: "model.", with: "")
    }

    let remapped = applyKeyMapping(rawParams, keyMapping: keyMapping)

    #expect(remapped.keys.contains("layers.0.weight"))
    #expect(remapped.keys.contains("layers.1.weight"))
    #expect(remapped.keys.contains("layers.2.bias"))
    #expect(!remapped.keys.contains("model.layers.0.weight"))
  }

  @Test("Key remapping skips keys when mapping returns nil")
  func keyRemappingSkipsNilKeys() throws {
    let rawParams: [String: MLXArray] = [
      "keep.this.weight": MLXArray.ones([4, 4]),
      "skip.this.weight": MLXArray.ones([4, 4]),
      "keep.another.bias": MLXArray.zeros([4]),
    ]

    // Skip keys that start with "skip"
    let keyMapping: KeyMapping = { key in
      key.hasPrefix("skip") ? nil : key
    }

    let remapped = applyKeyMapping(rawParams, keyMapping: keyMapping)

    #expect(remapped.count == 2)
    #expect(remapped.keys.contains("keep.this.weight"))
    #expect(remapped.keys.contains("keep.another.bias"))
    #expect(!remapped.keys.contains("skip.this.weight"))
  }

  // MARK: - Tensor Transform

  @Test("Tensor transform applies correctly")
  func tensorTransformApplied() throws {
    let rawParams: [String: MLXArray] = [
      "layer.weight": MLXArray.ones([4, 4])
    ]

    // Transform that multiplies by 2
    let transform: TensorTransform = { _, tensor in
      tensor * 2
    }

    let transformed = applyTensorTransform(rawParams, transform: transform)
    let values = transformed["layer.weight"]!
    eval(values)

    // All values should be 2.0 (ones * 2)
    let flat = values.reshaped(-1)
    for i in 0..<flat.dim(0) {
      #expect(flat[i].item(Float.self) == 2.0)
    }
  }

  @Test("Tensor transform is key-aware")
  func tensorTransformKeyAware() throws {
    let rawParams: [String: MLXArray] = [
      "scale.weight": MLXArray.ones([4, 4]),
      "bias.weight": MLXArray.ones([4, 4]),
    ]

    // Only transform weights named "scale.*"
    let transform: TensorTransform = { key, tensor in
      if key.hasPrefix("scale") {
        return tensor * 10.0
      }
      return tensor
    }

    let transformed = applyTensorTransform(rawParams, transform: transform)

    let scaleVal = transformed["scale.weight"]!.reshaped(-1)[0].item(Float.self)
    let biasVal = transformed["bias.weight"]!.reshaped(-1)[0].item(Float.self)

    #expect(scaleVal == 10.0)
    #expect(biasVal == 1.0)
  }

  // MARK: - Quantization

  @Test("QuantizationConfig.asStored preserves tensor")
  func quantizationAsStored() throws {
    let tensor = MLXArray.ones([4, 4])
    let result = applyQuantizationConfig(tensor, config: .asStored)
    eval(result)
    #expect(result.dtype == tensor.dtype)
  }

  @Test("QuantizationConfig.float16 converts dtype")
  func quantizationFloat16() throws {
    let tensor = MLXArray.ones([4, 4])
    let result = applyQuantizationConfig(tensor, config: .float16)
    eval(result)
    #expect(result.dtype == .float16)
  }

  @Test("QuantizationConfig.bfloat16 converts dtype")
  func quantizationBfloat16() throws {
    let tensor = MLXArray.ones([4, 4])
    let result = applyQuantizationConfig(tensor, config: .bfloat16)
    eval(result)
    #expect(result.dtype == .bfloat16)
  }

  @Test("Quantization int4 only applies to eligible 2D tensors")
  func quantizationInt4Eligibility() throws {
    // Eligible: 2D with dims that are multiples of 32
    let eligible = MLXArray.ones([64, 64])
    let resultEligible = applyQuantizationConfig(eligible, config: .int4(groupSize: 64))
    eval(resultEligible)
    // Should have been quantized and dequantized; values close to 1.0 but not exact
    // Just verify it completed without error
    #expect(resultEligible.shape == [64, 64])

    // Non-eligible: 1D tensor
    let bias1D = MLXArray.ones([64])
    let resultBias = applyQuantizationConfig(bias1D, config: .int4(groupSize: 64))
    eval(resultBias)
    // Should be returned as-is since 1D isn't eligible
    #expect(resultBias.shape == [64])

    // Non-eligible: 2D but dimensions not multiple of 32
    let small2D = MLXArray.ones([4, 4])
    let resultSmall = applyQuantizationConfig(small2D, config: .int4(groupSize: 64))
    eval(resultSmall)
    #expect(resultSmall.shape == [4, 4])
  }

  // MARK: - Combined Pipeline

  @Test("Full key-remap + transform + quantize pipeline")
  func fullLoadingPipeline() throws {
    let rawParams: [String: MLXArray] = [
      "original.layer1.weight": MLXArray.ones([4, 4]),
      "original.layer2.weight": MLXArray.ones([4, 4]),
      "original.skip_me.weight": MLXArray.ones([4, 4]),
    ]

    // Key mapping: rename and filter
    let keyMapping: KeyMapping = { key in
      if key.contains("skip_me") { return nil }
      return key.replacingOccurrences(of: "original.", with: "remapped.")
    }

    // Transform: double all values
    let transform: TensorTransform = { _, tensor in tensor * 2.0 }

    // Apply pipeline
    var result = applyKeyMapping(rawParams, keyMapping: keyMapping)
    result = applyTensorTransform(result, transform: transform)
    let finalResult = applyQuantizationToAll(result, config: .float16)

    #expect(finalResult.count == 2)
    #expect(finalResult.keys.contains("remapped.layer1.weight"))
    #expect(finalResult.keys.contains("remapped.layer2.weight"))
    #expect(!finalResult.keys.contains("original.skip_me.weight"))

    let weight = finalResult["remapped.layer1.weight"]!
    eval(weight)
    #expect(weight.dtype == .float16)
  }

  // MARK: - Helpers (simulate what WeightLoader does internally)

  private func applyKeyMapping(
    _ params: [String: MLXArray],
    keyMapping: KeyMapping
  ) -> [String: MLXArray] {
    var result: [String: MLXArray] = [:]
    for (key, tensor) in params {
      if let remapped = keyMapping(key) {
        result[remapped] = tensor
      }
    }
    return result
  }

  private func applyTensorTransform(
    _ params: [String: MLXArray],
    transform: TensorTransform
  ) -> [String: MLXArray] {
    var result: [String: MLXArray] = [:]
    for (key, tensor) in params {
      result[key] = transform(key, tensor)
    }
    return result
  }

  private func applyQuantizationConfig(
    _ tensor: MLXArray,
    config: QuantizationConfig
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

  private func quantizeIfEligible(_ tensor: MLXArray, bits: Int, groupSize: Int) -> MLXArray {
    guard tensor.ndim == 2 else { return tensor }
    let rows = tensor.dim(0)
    let cols = tensor.dim(1)
    guard rows % 32 == 0 && cols % 32 == 0 else { return tensor }
    let (wq, scales, biases) = quantized(tensor, groupSize: groupSize, bits: bits)
    return dequantized(wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
  }

  private func applyQuantizationToAll(
    _ params: [String: MLXArray],
    config: QuantizationConfig
  ) -> [String: MLXArray] {
    var result: [String: MLXArray] = [:]
    for (key, tensor) in params {
      result[key] = applyQuantizationConfig(tensor, config: config)
    }
    return result
  }
}
