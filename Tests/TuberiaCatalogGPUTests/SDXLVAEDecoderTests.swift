import Testing
@preconcurrency import MLX
@testable import TuberiaCatalog
import Tuberia

@Suite("SDXLVAEDecoder Shape Contract Tests")
struct SDXLVAEDecoderTests {

    // MARK: - Output Shape Contracts

    @Test("decode produces [B, H*8, W*8, 3] output shape")
    func decodeOutputShape() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        let latents = MLXArray.zeros([1, 8, 8, 4])
        let output = try decoder.decode(latents)

        #expect(output.data.shape == [1, 64, 64, 3])
    }

    @Test("8x spatial upscale holds across different latent sizes")
    func variousSpatialDimensions() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        let latents16 = MLXArray.zeros([1, 16, 16, 4])
        let output16 = try decoder.decode(latents16)
        #expect(output16.data.shape == [1, 128, 128, 3])

        let latents32 = MLXArray.zeros([1, 32, 32, 4])
        let output32 = try decoder.decode(latents32)
        #expect(output32.data.shape == [1, 256, 256, 3])
    }

    @Test("Batch decode works with B > 1")
    func batchDecode() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        let latents = MLXArray.zeros([2, 8, 8, 4])
        let output = try decoder.decode(latents)

        #expect(output.data.shape == [2, 64, 64, 3])
    }

    @Test("decode returns ImageDecoderMetadata with correct scalingFactor")
    func decodeMetadata() throws {
        let config = SDXLVAEDecoderConfiguration(scalingFactor: 0.13025)
        let decoder = try SDXLVAEDecoder(configuration: config)

        let latents = MLXArray.zeros([1, 8, 8, 4])
        let output = try decoder.decode(latents)

        guard let metadata = output.metadata as? ImageDecoderMetadata else {
            Issue.record("Expected ImageDecoderMetadata")
            return
        }
        #expect(abs(metadata.scalingFactor - 0.13025) < 0.0001)
    }

    // MARK: - Error Paths

    @Test("Wrong channel count throws decodingFailed")
    func wrongChannelsThrows() throws {
        let config = SDXLVAEDecoderConfiguration(latentChannels: 4)
        let decoder = try SDXLVAEDecoder(configuration: config)

        let badLatents = MLXArray.zeros([1, 8, 8, 8])

        #expect(throws: PipelineError.self) {
            try decoder.decode(badLatents)
        }
    }

    @Test("Wrong dimension count throws decodingFailed")
    func wrongDimensionsThrows() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        let badLatents = MLXArray.zeros([8, 8, 4])

        #expect(throws: PipelineError.self) {
            try decoder.decode(badLatents)
        }
    }

    // MARK: - Weight Lifecycle

    @Test("apply(weights:) loads, unload() clears")
    func weightLifecycle() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        #expect(!decoder.isLoaded)

        let weights = Tuberia.ModuleParameters(parameters: ["test.weight": MLXArray.zeros([4, 4])])
        try decoder.apply(weights: weights)
        #expect(decoder.isLoaded)

        decoder.unload()
        #expect(!decoder.isLoaded)
    }

    @Test("estimatedMemoryBytes is in expected range (~160 MB)")
    func estimatedMemory() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        let memoryBytes = decoder.estimatedMemoryBytes
        #expect(memoryBytes > 100_000_000, "Should be > 100 MB")
        #expect(memoryBytes < 300_000_000, "Should be < 300 MB")
    }

    // MARK: - Key Mapping

    @Test("keyMapping filters out encoder keys")
    func keyMappingFiltersEncoderKeys() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        let mapping = decoder.keyMapping

        #expect(mapping("decoder.conv_in.weight") != nil)
        #expect(mapping("post_quant_conv.weight") != nil)
        #expect(mapping("encoder.conv_in.weight") == nil)
        #expect(mapping("quant_conv.weight") == nil)
    }
}
