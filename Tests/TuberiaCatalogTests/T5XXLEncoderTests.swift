import Testing
@preconcurrency import MLX
@testable import TuberiaCatalog
import Tuberia

@Suite("T5XXLEncoder Tests")
struct T5XXLEncoderTests {

    @Test("T5XXLEncoder conforms to TextEncoder with correct Configuration")
    func protocolConformance() throws {
        let config = T5XXLEncoderConfiguration()
        let encoder = try T5XXLEncoder(configuration: config)
        _ = encoder
    }

    @Test("Default configuration has expected values")
    func defaultConfiguration() {
        let config = T5XXLEncoderConfiguration()
        #expect(config.componentId == "t5-xxl-encoder-int4")
        #expect(config.maxSequenceLength == 120)
        #expect(config.embeddingDim == 4096)
    }

    @Test("outputEmbeddingDim returns 4096")
    func embeddingDim() throws {
        let config = T5XXLEncoderConfiguration()
        let encoder = try T5XXLEncoder(configuration: config)
        #expect(encoder.outputEmbeddingDim == 4096)
    }

    @Test("Custom embedding dim is correctly reported")
    func customEmbeddingDim() throws {
        let config = T5XXLEncoderConfiguration(embeddingDim: 2048)
        let encoder = try T5XXLEncoder(configuration: config)
        #expect(encoder.outputEmbeddingDim == 2048)
    }

    @Test("maxSequenceLength returns config value")
    func maxSeqLength() throws {
        let config = T5XXLEncoderConfiguration(maxSequenceLength: 120)
        let encoder = try T5XXLEncoder(configuration: config)
        #expect(encoder.maxSequenceLength == 120)

        let config77 = T5XXLEncoderConfiguration(maxSequenceLength: 77)
        let encoder77 = try T5XXLEncoder(configuration: config77)
        #expect(encoder77.maxSequenceLength == 77)
    }

    @Test("encode produces correct output shape [1, seq, 4096]")
    func encodeOutputShape() throws {
        let config = T5XXLEncoderConfiguration(maxSequenceLength: 120, embeddingDim: 4096)
        let encoder = try T5XXLEncoder(configuration: config)

        let input = TextEncoderInput(text: "a photo of a cat", maxLength: 120)
        let output = try encoder.encode(input)

        // Shape: [1, seqLen, 4096]
        #expect(output.embeddings.shape.count == 3)
        #expect(output.embeddings.shape[0] == 1) // batch size
        #expect(output.embeddings.shape[1] == 120) // sequence length
        #expect(output.embeddings.shape[2] == 4096) // embedding dim
    }

    @Test("encode produces correct mask shape [1, seq]")
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

        // Short text should have some real tokens (1s) and rest padding (0s)
        let input = TextEncoderInput(text: "hi", maxLength: 120)
        let output = try encoder.encode(input)

        eval(output.mask)
        let maskSum = output.mask.sum().item(Float.self)

        // Should have at least 1 real token
        #expect(maskSum >= 1.0, "Mask should have at least 1 real token")
        // But not all tokens should be real (text is short)
        #expect(maskSum < 120.0, "Short text should not fill entire sequence")
    }

    @Test("maxLength parameter respects configured limit")
    func respectsMaxLength() throws {
        let config = T5XXLEncoderConfiguration(maxSequenceLength: 50)
        let encoder = try T5XXLEncoder(configuration: config)

        // Request longer than max should be clamped
        let input = TextEncoderInput(text: "test text", maxLength: 200)
        let output = try encoder.encode(input)

        // Should use min(200, 50) = 50
        #expect(output.embeddings.shape[1] == 50)
    }

    @Test("WeightedSegment: estimatedMemoryBytes returns ~1.2GB")
    func estimatedMemory() throws {
        let config = T5XXLEncoderConfiguration()
        let encoder = try T5XXLEncoder(configuration: config)

        let memoryBytes = encoder.estimatedMemoryBytes
        #expect(memoryBytes > 1_000_000_000, "Should be > 1 GB")
        #expect(memoryBytes < 2_000_000_000, "Should be < 2 GB")
    }

    @Test("WeightedSegment: apply(weights:) sets isLoaded = true")
    func applyWeights() throws {
        let config = T5XXLEncoderConfiguration()
        let encoder = try T5XXLEncoder(configuration: config)

        #expect(!encoder.isLoaded)

        let weights = Tuberia.ModuleParameters(parameters: ["shared.weight": MLXArray.zeros([4, 4])])
        try encoder.apply(weights: weights)

        #expect(encoder.isLoaded)
    }

    @Test("WeightedSegment: unload() sets isLoaded = false and clears weights")
    func unloadWeights() throws {
        let config = T5XXLEncoderConfiguration()
        let encoder = try T5XXLEncoder(configuration: config)

        let weights = Tuberia.ModuleParameters(parameters: ["shared.weight": MLXArray.zeros([4, 4])])
        try encoder.apply(weights: weights)
        #expect(encoder.isLoaded)

        encoder.unload()
        #expect(!encoder.isLoaded)
    }

    @Test("keyMapping keeps encoder keys and skips decoder keys")
    func keyMappingBehavior() throws {
        let config = T5XXLEncoderConfiguration()
        let encoder = try T5XXLEncoder(configuration: config)

        let mapping = encoder.keyMapping

        // Encoder keys should be kept
        #expect(mapping("encoder.block.0.layer.0.SelfAttention.q.weight") != nil)
        #expect(mapping("shared.weight") != nil)

        // Decoder keys should be skipped
        #expect(mapping("decoder.block.0.layer.0.SelfAttention.q.weight") == nil)
        #expect(mapping("lm_head.weight") == nil)
    }

    @Test("Empty text still produces valid output")
    func emptyText() throws {
        let config = T5XXLEncoderConfiguration(maxSequenceLength: 10)
        let encoder = try T5XXLEncoder(configuration: config)

        let input = TextEncoderInput(text: "", maxLength: 10)
        let output = try encoder.encode(input)

        #expect(output.embeddings.shape == [1, 10, 4096])
        #expect(output.mask.shape == [1, 10])
    }
}
