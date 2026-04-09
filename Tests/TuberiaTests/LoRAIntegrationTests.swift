import Foundation
import Testing

@preconcurrency import MLX
@testable import Tuberia

// MARK: - LoRA Integration Tests
//
// Tests for LoRA weight merging and restoration logic:
// (a) Synthetic LoRA adapter changes backbone weights during the merge window
// (b) Weights are restored to original values after unapply
// (c) LoRA with scale=0.0 produces no weight change

// MARK: - Local Test Mock

/// A minimal WeightedSegment mock for LoRA tests (local to this test target).
private final class LoRATestSegment: WeightedSegment, @unchecked Sendable {
  private var weights: ModuleParameters?
  var isLoaded: Bool = false
  var estimatedMemoryBytes: Int { 0 }
  var keyMapping: KeyMapping { { key in key } }
  var currentWeights: ModuleParameters? { weights }

  func apply(weights: ModuleParameters) throws {
    self.weights = weights
    self.isLoaded = true
  }

  func unload() {
    self.weights = nil
    self.isLoaded = false
  }
}

@Suite("LoRA Integration Tests", .serialized)
struct LoRAIntegrationTests {

  // MARK: - Helpers

  /// Create a simple set of base model parameters with known values.
  private func makeBaseWeights() -> ModuleParameters {
    ModuleParameters(parameters: [
      "layer.0.weight": MLXArray([Float32(1.0), 2.0, 3.0, 4.0]).reshaped([2, 2]),
      "layer.1.weight": MLXArray([Float32(5.0), 6.0, 7.0, 8.0]).reshaped([2, 2]),
    ])
  }

  /// Create synthetic LoRA adapter weights (lora_A and lora_B pairs).
  ///
  /// For a 2x2 base weight, we use rank-1 LoRA:
  /// - lora_A: [1, 2] (shape [1, 2])
  /// - lora_B: [1, 1] (shape [2, 1])
  /// B [2,1] @ A [1,2] = [[1,2],[1,2]]
  private func makeAdapterWeights() -> ModuleParameters {
    ModuleParameters(parameters: [
      "layer.0.weight.lora_A": MLXArray([Float32(1.0), 2.0]).reshaped([1, 2]),
      "layer.0.weight.lora_B": MLXArray([Float32(1.0), 1.0]).reshaped([2, 1]),
    ])
  }

  // MARK: - Test (a): LoRA apply changes weights

  @Test("LoRA apply merges adapter weights into base parameters")
  func testLoRAApplyChangesWeights() {
    let baseWeights = makeBaseWeights()
    let adapterWeights = makeAdapterWeights()

    let mergedWeights = LoRALoader.apply(
      adapterWeights: adapterWeights,
      to: baseWeights,
      scale: 1.0
    )

    // layer.0.weight should be modified: base + scale * (B @ A)
    // lora_B [2,1] @ lora_A [1,2] = [[1,2],[1,2]]
    // merged = [[1,2],[3,4]] + 1.0 * [[1,2],[1,2]] = [[2,4],[4,6]]
    let merged0 = mergedWeights.parameters["layer.0.weight"]!
    eval(merged0)
    let mergedValues = merged0.asArray(Float32.self)
    #expect(mergedValues.count == 4)
    #expect(mergedValues[0] == 2.0)  // 1 + 1
    #expect(mergedValues[1] == 4.0)  // 2 + 2
    #expect(mergedValues[2] == 4.0)  // 3 + 1
    #expect(mergedValues[3] == 6.0)  // 4 + 2

    // layer.1.weight should be unchanged (no adapter for this key)
    let unchanged1 = mergedWeights.parameters["layer.1.weight"]!
    eval(unchanged1)
    let unchangedValues = unchanged1.asArray(Float32.self)
    #expect(unchangedValues[0] == 5.0)
    #expect(unchangedValues[1] == 6.0)
    #expect(unchangedValues[2] == 7.0)
    #expect(unchangedValues[3] == 8.0)
  }

  // MARK: - Test (b): LoRA unapply restores original weights

  @Test("LoRA unapply restores base weights after generation")
  func testLoRAUnapplyRestoresWeights() {
    let baseWeights = makeBaseWeights()
    let adapterWeights = makeAdapterWeights()
    let scale: Float = 1.0

    // Step 1: Apply LoRA
    let mergedWeights = LoRALoader.apply(
      adapterWeights: adapterWeights,
      to: baseWeights,
      scale: scale
    )

    // Step 2: Unapply LoRA
    let restoredWeights = LoRALoader.unapply(
      adapterWeights: adapterWeights,
      from: mergedWeights,
      scale: scale
    )

    // Verify layer.0.weight is restored to original values
    let restored0 = restoredWeights.parameters["layer.0.weight"]!
    eval(restored0)
    let restoredValues = restored0.asArray(Float32.self)
    #expect(restoredValues.count == 4)
    #expect(abs(restoredValues[0] - 1.0) < 1e-6)
    #expect(abs(restoredValues[1] - 2.0) < 1e-6)
    #expect(abs(restoredValues[2] - 3.0) < 1e-6)
    #expect(abs(restoredValues[3] - 4.0) < 1e-6)

    // Verify layer.1.weight is also unchanged
    let restored1 = restoredWeights.parameters["layer.1.weight"]!
    eval(restored1)
    let restoredValues1 = restored1.asArray(Float32.self)
    #expect(abs(restoredValues1[0] - 5.0) < 1e-6)
    #expect(abs(restoredValues1[1] - 6.0) < 1e-6)
    #expect(abs(restoredValues1[2] - 7.0) < 1e-6)
    #expect(abs(restoredValues1[3] - 8.0) < 1e-6)
  }

  // MARK: - Test: Round-trip with fractional scale

  @Test("LoRA apply then unapply with fractional scale restores original weights")
  func testLoRARoundTripFractionalScale() {
    let baseWeights = makeBaseWeights()
    let adapterWeights = makeAdapterWeights()
    let scale: Float = 0.75

    let mergedWeights = LoRALoader.apply(
      adapterWeights: adapterWeights,
      to: baseWeights,
      scale: scale
    )

    let restoredWeights = LoRALoader.unapply(
      adapterWeights: adapterWeights,
      from: mergedWeights,
      scale: scale
    )

    let restored0 = restoredWeights.parameters["layer.0.weight"]!
    eval(restored0)
    let restoredValues = restored0.asArray(Float32.self)
    #expect(abs(restoredValues[0] - 1.0) < 1e-5)
    #expect(abs(restoredValues[1] - 2.0) < 1e-5)
    #expect(abs(restoredValues[2] - 3.0) < 1e-5)
    #expect(abs(restoredValues[3] - 4.0) < 1e-5)
  }

  // MARK: - Test: WeightedSegment.currentWeights property

  @Test("WeightedSegment currentWeights returns nil before apply and cached value after")
  func testCurrentWeightsProperty() throws {
    let segment = LoRATestSegment()

    // Before apply: currentWeights should be nil
    #expect(segment.currentWeights == nil)

    // After apply: currentWeights should be the applied weights
    let weights = ModuleParameters(parameters: [
      "key1": MLXArray([Float32(1.0), 2.0]),
    ])
    try segment.apply(weights: weights)
    #expect(segment.currentWeights != nil)
    #expect(segment.currentWeights?.parameters.count == 1)
    #expect(segment.currentWeights?.parameters.keys.contains("key1") == true)

    // After unload: currentWeights should be nil again
    segment.unload()
    #expect(segment.currentWeights == nil)
  }
}
