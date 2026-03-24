import Testing
@preconcurrency import MLX
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
        let baseWeight = make2D([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ], rows: 4, cols: 4)

        // LoRA A: 2x4 (rank 2)
        let loraA = make2D([
            1, 0, 0, 0,
            0, 1, 0, 0,
        ], rows: 2, cols: 4)

        // LoRA B: 4x2
        let loraB = make2D([
            0.5, 0.0,
            0.0, 0.5,
            0.0, 0.0,
            0.0, 0.0,
        ], rows: 4, cols: 2)

        let baseParams = ModuleParameters(parameters: [
            "layer.weight": baseWeight,
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
            "layer.weight": baseWeight,
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
            "layer.weight": baseWeight,
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
        let baseWeight = make2D([
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        ], rows: 4, cols: 4)

        let loraA = make2D([1, 1, 1, 1], rows: 1, cols: 4)
        let loraB = make2D([1, 1, 1, 1], rows: 4, cols: 1)

        let scale: Float = 0.75

        let baseParams = ModuleParameters(parameters: [
            "layer.weight": baseWeight,
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
            "block.0.attn.q_proj.weight": MLXArray.ones([4, 4]),
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
            "layer.weight": baseWeight,
        ])

        // Only provide lora_A, no lora_B
        let adapterParams = ModuleParameters(parameters: [
            "layer.weight.lora_A": MLXArray.ones([2, 4]),
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
            "layer.weight": baseWeight,
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
