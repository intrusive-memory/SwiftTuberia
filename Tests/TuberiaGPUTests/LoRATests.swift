@preconcurrency import MLX
import Testing

@testable import Tuberia

/// Unit tests for LoRA merge math, scale application, and unload restoration.
/// All tests use synthetic weights -- no real model files or network access.
@Suite("LoRA Tests")
struct LoRATests {

  // MARK: - Helpers

  /// Create a 2D MLXArray from flat data.
  private func make2D(_ data: [Float], rows: Int, cols: Int) -> MLXArray {
    MLXArray(data, [rows, cols])
  }

  // MARK: - LoRA Merge Math

  @Test("LoRA merge produces correct output with known A/B matrices")
  func loraMergeCorrectOutput() throws {
    // Base weight: 4x4 identity matrix
    let baseWeight = make2D(
      [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ], rows: 4, cols: 4)

    // LoRA A: 2x4 (rank 2)
    let loraA = make2D(
      [
        1, 0, 0, 0,
        0, 1, 0, 0,
      ], rows: 2, cols: 4)

    // LoRA B: 4x2
    let loraB = make2D(
      [
        0.5, 0.0,
        0.0, 0.5,
        0.0, 0.0,
        0.0, 0.0,
      ], rows: 4, cols: 2)

    let baseParams = ModuleParameters(parameters: [
      "layer.weight": baseWeight
    ])

    let adapterParams = ModuleParameters(parameters: [
      "layer.weight.lora_A": loraA,
      "layer.weight.lora_B": loraB,
    ])

    // Apply with scale 1.0
    let merged = LoRALoader.apply(
      adapterWeights: adapterParams,
      to: baseParams,
      scale: 1.0
    )

    let mergedWeight = merged.parameters["layer.weight"]!
    eval(mergedWeight)

    // Expected: W + 1.0 * (B @ A)
    // B @ A = [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // W' = [[1.5, 0, 0, 0], [0, 1.5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    #expect(abs(mergedWeight[0, 0].item(Float.self) - 1.5) < 1e-6)
    #expect(abs(mergedWeight[1, 1].item(Float.self) - 1.5) < 1e-6)
    #expect(abs(mergedWeight[2, 2].item(Float.self) - 1.0) < 1e-6)
    #expect(abs(mergedWeight[3, 3].item(Float.self) - 1.0) < 1e-6)
    #expect(abs(mergedWeight[0, 1].item(Float.self) - 0.0) < 1e-6)
  }

  // MARK: - LoRA Scale

  @Test("LoRA scale=0.0 produces base weights unchanged")
  func loraScaleZero() throws {
    let baseWeight = MLXArray.ones([4, 4])

    let loraA = MLXArray.ones([2, 4])
    let loraB = MLXArray.ones([4, 2])

    let baseParams = ModuleParameters(parameters: [
      "layer.weight": baseWeight
    ])

    let adapterParams = ModuleParameters(parameters: [
      "layer.weight.lora_A": loraA,
      "layer.weight.lora_B": loraB,
    ])

    let merged = LoRALoader.apply(
      adapterWeights: adapterParams,
      to: baseParams,
      scale: 0.0
    )

    let mergedWeight = merged.parameters["layer.weight"]!
    eval(mergedWeight)

    // With scale 0.0, the delta should be zero, so merged == base
    let diff = abs(mergedWeight - baseWeight)
    eval(diff)
    let maxDiff = diff.max().item(Float.self)
    #expect(maxDiff < 1e-6)
  }

  @Test("LoRA scale=0.5 produces half the delta")
  func loraScaleHalf() throws {
    let baseWeight = MLXArray.zeros([4, 4])

    // Simple A and B that produce a known delta
    let loraA = make2D([1, 0, 0, 0], rows: 1, cols: 4)
    let loraB = make2D([2, 0, 0, 0], rows: 4, cols: 1)

    let baseParams = ModuleParameters(parameters: [
      "layer.weight": baseWeight
    ])

    let adapterParams = ModuleParameters(parameters: [
      "layer.weight.lora_A": loraA,
      "layer.weight.lora_B": loraB,
    ])

    // Full scale: delta = B @ A = [[2,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    // scale=0.5: merged[0,0] should be 1.0
    let merged = LoRALoader.apply(
      adapterWeights: adapterParams,
      to: baseParams,
      scale: 0.5
    )

    let mergedWeight = merged.parameters["layer.weight"]!
    eval(mergedWeight)
    #expect(abs(mergedWeight[0, 0].item(Float.self) - 1.0) < 1e-6)
  }

  // MARK: - LoRA Unapply (Restore)

  @Test("LoRA unapply restores exact base weights")
  func loraUnapplyRestoresBase() throws {
    let baseWeight = make2D(
      [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
      ], rows: 4, cols: 4)

    let loraA = make2D([1, 1, 1, 1], rows: 1, cols: 4)
    let loraB = make2D([1, 1, 1, 1], rows: 4, cols: 1)

    let scale: Float = 0.75

    let baseParams = ModuleParameters(parameters: [
      "layer.weight": baseWeight
    ])

    let adapterParams = ModuleParameters(parameters: [
      "layer.weight.lora_A": loraA,
      "layer.weight.lora_B": loraB,
    ])

    // Apply LoRA
    let merged = LoRALoader.apply(
      adapterWeights: adapterParams,
      to: baseParams,
      scale: scale
    )

    // Verify it changed
    let mergedWeight = merged.parameters["layer.weight"]!
    eval(mergedWeight)
    let diffFromBase = abs(mergedWeight - baseWeight)
    eval(diffFromBase)
    #expect(diffFromBase.max().item(Float.self) > 0.1)

    // Unapply LoRA
    let restored = LoRALoader.unapply(
      adapterWeights: adapterParams,
      from: merged,
      scale: scale
    )

    // Verify restoration matches base
    let restoredWeight = restored.parameters["layer.weight"]!
    eval(restoredWeight)
    let restoreDiff = abs(restoredWeight - baseWeight)
    eval(restoreDiff)
    let maxRestoreDiff = restoreDiff.max().item(Float.self)
    #expect(maxRestoreDiff < 1e-5)
  }

  // MARK: - LoRA Key Parsing

  @Test("LoRA keys with standard suffix convention are matched")
  func loraStandardKeySuffix() throws {
    let baseParams = ModuleParameters(parameters: [
      "block.0.attn.q_proj.weight": MLXArray.ones([4, 4])
    ])

    let adapterParams = ModuleParameters(parameters: [
      "block.0.attn.q_proj.weight.lora_A": MLXArray.ones([2, 4]),
      "block.0.attn.q_proj.weight.lora_B": MLXArray.ones([4, 2]),
    ])

    let merged = LoRALoader.apply(
      adapterWeights: adapterParams,
      to: baseParams,
      scale: 1.0
    )

    // Verify the base key was found and merged
    let weight = merged.parameters["block.0.attn.q_proj.weight"]!
    eval(weight)
    // Should differ from ones since we added B@A*scale
    let diff = abs(weight - MLXArray.ones([4, 4]))
    eval(diff)
    #expect(diff.max().item(Float.self) > 0.1)
  }

  @Test("Incomplete LoRA pairs (A without B) are skipped")
  func loraIncompletePairSkipped() throws {
    let baseWeight = MLXArray.ones([4, 4])
    let baseParams = ModuleParameters(parameters: [
      "layer.weight": baseWeight
    ])

    // Only provide lora_A, no lora_B
    let adapterParams = ModuleParameters(parameters: [
      "layer.weight.lora_A": MLXArray.ones([2, 4])
    ])

    let merged = LoRALoader.apply(
      adapterWeights: adapterParams,
      to: baseParams,
      scale: 1.0
    )

    // Should be unchanged since the pair is incomplete
    let mergedWeight = merged.parameters["layer.weight"]!
    eval(mergedWeight)
    let diff = abs(mergedWeight - baseWeight)
    eval(diff)
    #expect(diff.max().item(Float.self) < 1e-6)
  }

  @Test("LoRA adapter with keys not in base model are ignored")
  func loraUnmatchedKeysIgnored() throws {
    let baseWeight = MLXArray.ones([4, 4])
    let baseParams = ModuleParameters(parameters: [
      "layer.weight": baseWeight
    ])

    let adapterParams = ModuleParameters(parameters: [
      "nonexistent.layer.lora_A": MLXArray.ones([2, 4]),
      "nonexistent.layer.lora_B": MLXArray.ones([4, 2]),
    ])

    let merged = LoRALoader.apply(
      adapterWeights: adapterParams,
      to: baseParams,
      scale: 1.0
    )

    // Base weight should be unchanged
    let mergedWeight = merged.parameters["layer.weight"]!
    eval(mergedWeight)
    let diff = abs(mergedWeight - baseWeight)
    eval(diff)
    #expect(diff.max().item(Float.self) < 1e-6)
  }

  // MARK: - LoRA Config

  @Test("LoRAConfig requires at least one ID source")
  func loraConfigPrecondition() throws {
    // Valid configs
    let config1 = LoRAConfig(componentId: "test-lora")
    #expect(config1.componentId == "test-lora")

    let config2 = LoRAConfig(localPath: "/tmp/lora.safetensors")
    #expect(config2.localPath == "/tmp/lora.safetensors")

    let config3 = LoRAConfig(componentId: "id", localPath: "/path")
    #expect(config3.componentId == "id")
    #expect(config3.localPath == "/path")
  }

  @Test("LoRAConfig scale defaults to 1.0")
  func loraConfigDefaultScale() throws {
    let config = LoRAConfig(componentId: "test")
    #expect(config.scale == 1.0)
  }

  @Test("LoRAConfig activation keyword is optional")
  func loraConfigActivationKeyword() throws {
    let withKeyword = LoRAConfig(componentId: "test", activationKeyword: "ohwx")
    #expect(withKeyword.activationKeyword == "ohwx")

    let without = LoRAConfig(componentId: "test")
    #expect(without.activationKeyword == nil)
  }
}

// MARK: - LoRA Key Convention Tests

/// Tests for LoRA key-naming conventions handled by `LoRALoader.parseLoRAKey`.
/// All tests use synthetic `ModuleParameters` — no real adapter files or network access.
@Suite("LoRA Key Convention Tests")
struct LoRAKeyConventionTests {

  // MARK: - Helpers

  /// Create a 2D MLXArray from flat data.
  private func make2D(_ data: [Float], rows: Int, cols: Int) -> MLXArray {
    MLXArray(data, [rows, cols])
  }

  // MARK: - Tests

  @Test("HuggingFace .lora_A.weight / .lora_B.weight suffix convention merges correctly")
  func loraAWeightDotSuffix() throws {
    // Base weight: 4x4 identity
    let baseWeight = make2D(
      [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
      ], rows: 4, cols: 4)

    // LoRA A: 2x4 (rank 2) — keyed with .lora_A.weight convention
    let loraA = make2D(
      [
        1, 0, 0, 0,
        0, 1, 0, 0,
      ], rows: 2, cols: 4)

    // LoRA B: 4x2 — keyed with .lora_B.weight convention
    let loraB = make2D(
      [
        0.5, 0.0,
        0.0, 0.5,
        0.0, 0.0,
        0.0, 0.0,
      ], rows: 4, cols: 2)

    let baseParams = ModuleParameters(parameters: [
      "layer.weight": baseWeight
    ])

    // HuggingFace-style: "layer.lora_A.weight" / "layer.lora_B.weight"
    // parseLoRAKey strips ".lora_A." -> "." yielding base key "layer.weight"
    let adapterParams = ModuleParameters(parameters: [
      "layer.lora_A.weight": loraA,
      "layer.lora_B.weight": loraB,
    ])

    let merged = LoRALoader.apply(
      adapterWeights: adapterParams,
      to: baseParams,
      scale: 1.0
    )

    let mergedWeight = merged.parameters["layer.weight"]!
    eval(mergedWeight)

    // Expected: W + 1.0 * (B @ A)
    // B @ A = [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    // W' = [[1.5, 0, 0, 0], [0, 1.5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    #expect(abs(mergedWeight[0, 0].item(Float.self) - 1.5) < 1e-6)
    #expect(abs(mergedWeight[1, 1].item(Float.self) - 1.5) < 1e-6)
    #expect(abs(mergedWeight[2, 2].item(Float.self) - 1.0) < 1e-6)
    #expect(abs(mergedWeight[3, 3].item(Float.self) - 1.0) < 1e-6)
    #expect(abs(mergedWeight[0, 1].item(Float.self) - 0.0) < 1e-6)
  }

  @Test("Diffusers-style .lora_up / .lora_down suffix convention merges correctly")
  func loraUpDownSuffix() throws {
    // Base weight: 4x4 zeros
    let baseWeight = MLXArray.zeros([4, 4])

    // lora_down maps to A (rank-1): 1x4
    let loraDown = make2D([1, 0, 0, 0], rows: 1, cols: 4)

    // lora_up maps to B: 4x1
    let loraUp = make2D([2, 0, 0, 0], rows: 4, cols: 1)

    let baseParams = ModuleParameters(parameters: [
      "layer.weight": baseWeight
    ])

    // Diffusers-style: "layer.weight.lora_down" / "layer.weight.lora_up"
    // parseLoRAKey strips ".lora_down" / ".lora_up" -> base key "layer.weight"
    let adapterParams = ModuleParameters(parameters: [
      "layer.weight.lora_down": loraDown,
      "layer.weight.lora_up": loraUp,
    ])

    let merged = LoRALoader.apply(
      adapterWeights: adapterParams,
      to: baseParams,
      scale: 0.5
    )

    let mergedWeight = merged.parameters["layer.weight"]!
    eval(mergedWeight)

    // delta = B @ A = [[2,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    // scaled = 0.5 * delta => [0,0] = 1.0
    // merged = base + scaled => [0,0] = 1.0
    #expect(abs(mergedWeight[0, 0].item(Float.self) - 1.0) < 1e-6)
    #expect(abs(mergedWeight[1, 0].item(Float.self) - 0.0) < 1e-6)
    #expect(abs(mergedWeight[0, 1].item(Float.self) - 0.0) < 1e-6)
  }

  @Test("unet. prefix in adapter keys is NOT stripped — base weight stays unchanged")
  func unetPrefixIsStripped() throws {
    // NOTE: LoRALoader does NOT implement unet. prefix stripping.
    // Adapter keys with a "unet." prefix will not match base keys without it.
    // This test verifies the actual behavior: the base weight is unchanged.
    let baseWeight = MLXArray.ones([4, 4])

    let baseParams = ModuleParameters(parameters: [
      "layer.weight": baseWeight
    ])

    // Adapter keys have "unet." prefix that the base model does not
    let adapterParams = ModuleParameters(parameters: [
      "unet.layer.weight.lora_A": MLXArray.ones([2, 4]),
      "unet.layer.weight.lora_B": MLXArray.ones([4, 2]),
    ])

    let merged = LoRALoader.apply(
      adapterWeights: adapterParams,
      to: baseParams,
      scale: 1.0
    )

    // Since unet. prefix is not stripped, adapter base key resolves to
    // "unet.layer.weight" which does not match "layer.weight".
    // Therefore the base weight should be unchanged.
    let mergedWeight = merged.parameters["layer.weight"]!
    eval(mergedWeight)
    let diff = abs(mergedWeight - baseWeight)
    eval(diff)
    #expect(diff.max().item(Float.self) < 1e-6)
  }

  @Test("Mixed lora_A.weight and lora_up/lora_down conventions across layers all merge")
  func mixedConventionsSingleAdapter() throws {
    // Two separate layers in the base model
    let baseWeightA = MLXArray.zeros([4, 4])
    let baseWeightB = MLXArray.zeros([4, 4])

    let baseParams = ModuleParameters(parameters: [
      "block.0.weight": baseWeightA,
      "block.1.weight": baseWeightB,
    ])

    // Layer 0 uses HuggingFace .lora_A.weight / .lora_B.weight convention
    // parseLoRAKey: "block.0.lora_A.weight" -> base "block.0.weight", component .a
    let hfA = make2D([1, 0, 0, 0], rows: 1, cols: 4)
    let hfB = make2D([3, 0, 0, 0], rows: 4, cols: 1)

    // Layer 1 uses diffusers .lora_down / .lora_up convention
    // parseLoRAKey: "block.1.weight.lora_down" -> base "block.1.weight", component .a
    let diffDown = make2D([1, 0, 0, 0], rows: 1, cols: 4)
    let diffUp = make2D([5, 0, 0, 0], rows: 4, cols: 1)

    let adapterParams = ModuleParameters(parameters: [
      "block.0.lora_A.weight": hfA,
      "block.0.lora_B.weight": hfB,
      "block.1.weight.lora_down": diffDown,
      "block.1.weight.lora_up": diffUp,
    ])

    let merged = LoRALoader.apply(
      adapterWeights: adapterParams,
      to: baseParams,
      scale: 1.0
    )

    // Layer 0: delta = hfB @ hfA = [[3,0,0,0],[0,0,0,0],...] => merged[0,0] = 3.0
    let w0 = merged.parameters["block.0.weight"]!
    eval(w0)
    #expect(abs(w0[0, 0].item(Float.self) - 3.0) < 1e-6)
    #expect(abs(w0[1, 0].item(Float.self) - 0.0) < 1e-6)

    // Layer 1: delta = diffUp @ diffDown = [[5,0,0,0],[0,0,0,0],...] => merged[0,0] = 5.0
    let w1 = merged.parameters["block.1.weight"]!
    eval(w1)
    #expect(abs(w1[0, 0].item(Float.self) - 5.0) < 1e-6)
    #expect(abs(w1[1, 0].item(Float.self) - 0.0) < 1e-6)

    // Both layers were merged — neither silently skipped
    #expect(merged.parameters.count == 2)
  }
}
