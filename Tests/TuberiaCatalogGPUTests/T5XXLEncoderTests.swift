import Testing
@preconcurrency import MLX
@testable import TuberiaCatalog
import Tuberia

@Suite("T5XXLEncoder Shape Contract Tests")
struct T5XXLEncoderTests {

    // MARK: - Output Shape Contracts

    @Test("encode produces [1, seq, embeddingDim] output shape")
    func encodeOutputShape() throws {
        let config = T5XXLEncoderConfiguration(maxSequenceLength: 120, embeddingDim: 4096)
        let encoder = try T5XXLEncoder(configuration: config)

        let input = TextEncoderInput(text: "a photo of a cat", maxLength: 120)
        let output = try encoder.encode(input)

        #expect(output.embeddings.shape == [1, 120, 4096])
    }

    @Test("encode produces mask shape [1, seq]")
    func encodeMaskShape() throws {
        let config = T5XXLEncoderConfiguration(maxSequenceLength: 120)
        let encoder = try T5XXLEncoder(configuration: config)

        let input = TextEncoderInput(text: "hello", maxLength: 120)
        let output = try encoder.encode(input)

        #expect(output.mask.shape == [1, 120])
    }

    @Test("Mask has 1s for real tokens and 0s for padding")
    func maskValues() throws {
        let config = T5XXLEncoderConfiguration(maxSequenceLength: 120)
        let encoder = try T5XXLEncoder(configuration: config)

        let input = TextEncoderInput(text: "hi", maxLength: 120)
        let output = try encoder.encode(input)

        eval(output.mask)
        let maskSum = output.mask.sum().item(Float.self)

        #expect(maskSum >= 1.0, "Mask should have at least 1 real token")
        #expect(maskSum < 120.0, "Short text should not fill entire sequence")
    }

    @Test("maxLength clamps to configured maxSequenceLength")
    func respectsMaxLength() throws {
        let config = T5XXLEncoderConfiguration(maxSequenceLength: 50)
        let encoder = try T5XXLEncoder(configuration: config)

        let input = TextEncoderInput(text: "test text", maxLength: 200)
        let output = try encoder.encode(input)

        #expect(output.embeddings.shape[1] == 50)
    }

    @Test("Empty text produces valid output shapes")
    func emptyText() throws {
        let config = T5XXLEncoderConfiguration(maxSequenceLength: 10)
        let encoder = try T5XXLEncoder(configuration: config)

        let input = TextEncoderInput(text: "", maxLength: 10)
        let output = try encoder.encode(input)

        #expect(output.embeddings.shape == [1, 10, 4096])
        #expect(output.mask.shape == [1, 10])
    }

    // MARK: - Weight Lifecycle

    @Test("apply(weights:) loads, unload() clears")
    func weightLifecycle() throws {
        let config = T5XXLEncoderConfiguration()
        let encoder = try T5XXLEncoder(configuration: config)

        #expect(!encoder.isLoaded)

        let weights = Tuberia.ModuleParameters(parameters: ["shared.weight": MLXArray.zeros([4, 4])])
        try encoder.apply(weights: weights)
        #expect(encoder.isLoaded)

        encoder.unload()
        #expect(!encoder.isLoaded)
    }

    @Test("estimatedMemoryBytes is in expected range (~1.2 GB)")
    func estimatedMemory() throws {
        let config = T5XXLEncoderConfiguration()
        let encoder = try T5XXLEncoder(configuration: config)

        let memoryBytes = encoder.estimatedMemoryBytes
        #expect(memoryBytes > 1_000_000_000, "Should be > 1 GB")
        #expect(memoryBytes < 2_000_000_000, "Should be < 2 GB")
    }

    // MARK: - Key Mapping

    @Test("keyMapping keeps encoder keys and skips decoder keys")
    func keyMappingBehavior() throws {
        let config = T5XXLEncoderConfiguration()
        let encoder = try T5XXLEncoder(configuration: config)

        let mapping = encoder.keyMapping

        #expect(mapping("encoder.block.0.layer.0.SelfAttention.q.weight") != nil)
        #expect(mapping("shared.weight") != nil)
        #expect(mapping("decoder.block.0.layer.0.SelfAttention.q.weight") == nil)
        #expect(mapping("lm_head.weight") == nil)
    }
}
