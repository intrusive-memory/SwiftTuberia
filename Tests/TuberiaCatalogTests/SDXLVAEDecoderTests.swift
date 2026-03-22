import Testing
@preconcurrency import MLX
@testable import TuberiaCatalog
import Tuberia

@Suite("SDXLVAEDecoder Tests")
struct SDXLVAEDecoderTests {

    @Test("SDXLVAEDecoder conforms to Decoder with correct Configuration")
    func protocolConformance() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)
        _ = decoder
    }

    @Test("Default configuration has expected values")
    func defaultConfiguration() {
        let config = SDXLVAEDecoderConfiguration()
        #expect(config.componentId == "sdxl-vae-decoder-fp16")
        #expect(config.latentChannels == 4)
        #expect(abs(config.scalingFactor - 0.13025) < 0.0001)
    }

    @Test("expectedInputChannels returns latentChannels from config")
    func inputChannels() throws {
        let config = SDXLVAEDecoderConfiguration(latentChannels: 4)
        let decoder = try SDXLVAEDecoder(configuration: config)
        #expect(decoder.expectedInputChannels == 4)

        let config8 = SDXLVAEDecoderConfiguration(latentChannels: 8)
        let decoder8 = try SDXLVAEDecoder(configuration: config8)
        #expect(decoder8.expectedInputChannels == 8)
    }

    @Test("scalingFactor returns value from config")
    func scalingFactorMatch() throws {
        let config = SDXLVAEDecoderConfiguration(scalingFactor: 0.18215)
        let decoder = try SDXLVAEDecoder(configuration: config)
        #expect(abs(decoder.scalingFactor - 0.18215) < 0.0001)
    }

    @Test("decode produces correct output shape [B, H*8, W*8, 3]")
    func decodeOutputShape() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        // Input: [1, 8, 8, 4] -> Output: [1, 64, 64, 3]
        let latents = MLXArray.zeros([1, 8, 8, 4])
        let output = try decoder.decode(latents)

        #expect(output.data.shape == [1, 64, 64, 3])
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

    @Test("decode with various spatial dimensions maintains 8x upscale")
    func variousSpatialDimensions() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        // 16x16 latents -> 128x128 pixels
        let latents16 = MLXArray.zeros([1, 16, 16, 4])
        let output16 = try decoder.decode(latents16)
        #expect(output16.data.shape == [1, 128, 128, 3])

        // 32x32 latents -> 256x256 pixels
        let latents32 = MLXArray.zeros([1, 32, 32, 4])
        let output32 = try decoder.decode(latents32)
        #expect(output32.data.shape == [1, 256, 256, 3])
    }

    @Test("decode with wrong channel count throws decodingFailed")
    func wrongChannelsThrows() throws {
        let config = SDXLVAEDecoderConfiguration(latentChannels: 4)
        let decoder = try SDXLVAEDecoder(configuration: config)

        // 8 channels instead of 4
        let badLatents = MLXArray.zeros([1, 8, 8, 8])

        #expect(throws: PipelineError.self) {
            try decoder.decode(badLatents)
        }
    }

    @Test("decode with wrong dimension count throws decodingFailed")
    func wrongDimensionsThrows() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        // 3D instead of 4D
        let badLatents = MLXArray.zeros([8, 8, 4])

        #expect(throws: PipelineError.self) {
            try decoder.decode(badLatents)
        }
    }

    @Test("WeightedSegment: estimatedMemoryBytes returns ~160MB")
    func estimatedMemory() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        let memoryBytes = decoder.estimatedMemoryBytes
        #expect(memoryBytes > 100_000_000, "Should be > 100 MB")
        #expect(memoryBytes < 300_000_000, "Should be < 300 MB")
    }

    @Test("WeightedSegment: apply(weights:) sets isLoaded = true")
    func applyWeights() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        #expect(!decoder.isLoaded)

        let weights = Tuberia.ModuleParameters(parameters: ["test.weight": MLXArray.zeros([4, 4])])
        try decoder.apply(weights: weights)

        #expect(decoder.isLoaded)
    }

    @Test("WeightedSegment: unload() sets isLoaded = false")
    func unloadWeights() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        let weights = Tuberia.ModuleParameters(parameters: ["test.weight": MLXArray.zeros([4, 4])])
        try decoder.apply(weights: weights)
        #expect(decoder.isLoaded)

        decoder.unload()
        #expect(!decoder.isLoaded)
    }

    @Test("keyMapping filters out encoder keys")
    func keyMappingFiltersEncoderKeys() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        let mapping = decoder.keyMapping

        // Decoder keys should be kept
        #expect(mapping("decoder.conv_in.weight") != nil)
        #expect(mapping("post_quant_conv.weight") != nil)

        // Encoder keys should be skipped
        #expect(mapping("encoder.conv_in.weight") == nil)
        #expect(mapping("quant_conv.weight") == nil)
    }

    @Test("Batch decode works with B > 1")
    func batchDecode() throws {
        let config = SDXLVAEDecoderConfiguration()
        let decoder = try SDXLVAEDecoder(configuration: config)

        // Batch of 2
        let latents = MLXArray.zeros([2, 8, 8, 4])
        let output = try decoder.decode(latents)

        #expect(output.data.shape == [2, 64, 64, 3])
    }
}
