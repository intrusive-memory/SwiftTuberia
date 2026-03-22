import Testing
@preconcurrency import MLX
@testable import Tuberia

/// Tests validating shape contracts at assembly time.
/// These verify that incompatible components produce clear PipelineError.incompatibleComponents errors.
@Suite("Shape Validation Tests")
struct ShapeValidationTests {

    // MARK: - Compatible Assembly

    @Test("Compatible encoder-backbone assembles successfully")
    func compatibleAssembly() throws {
        let recipe = StandardMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
            schedulerConfig: .init(),
            backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 120),
            decoderConfig: .init(inputChannels: 4),
            rendererConfig: ()
        )
        let pipeline = try DiffusionPipeline(recipe: recipe)
        #expect(pipeline.memoryRequirement.peakMemoryBytes > 0)
    }

    // MARK: - Embedding Dimension Mismatch

    @Test("Incompatible embedding dimension throws with clear message")
    func embeddingDimMismatch() throws {
        let recipe = StandardMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
            schedulerConfig: .init(),
            backboneConfig: .init(conditioningDim: 768, latentChannels: 4, maxSequenceLength: 120),
            decoderConfig: .init(inputChannels: 4),
            rendererConfig: ()
        )

        #expect(throws: PipelineError.self) {
            _ = try DiffusionPipeline(recipe: recipe)
        }

        do {
            _ = try DiffusionPipeline(recipe: recipe)
            Issue.record("Expected PipelineError.incompatibleComponents")
        } catch let error as PipelineError {
            switch error {
            case .incompatibleComponents(_, _, let reason):
                #expect(reason.contains("embedding dimension") || reason.contains("Embedding dimension"))
            default:
                Issue.record("Expected incompatibleComponents, got \(error)")
            }
        }
    }

    // MARK: - Sequence Length Mismatch

    @Test("Incompatible sequence length throws with clear message")
    func sequenceLengthMismatch() throws {
        let recipe = StandardMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
            schedulerConfig: .init(),
            backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 77),
            decoderConfig: .init(inputChannels: 4),
            rendererConfig: ()
        )

        do {
            _ = try DiffusionPipeline(recipe: recipe)
            Issue.record("Expected PipelineError.incompatibleComponents")
        } catch let error as PipelineError {
            switch error {
            case .incompatibleComponents(_, _, let reason):
                #expect(reason.contains("sequence length") || reason.contains("Sequence length"))
            default:
                Issue.record("Expected incompatibleComponents, got \(error)")
            }
        }
    }

    // MARK: - Latent Channel Mismatch

    @Test("Incompatible backbone-decoder channels throws with clear message")
    func latentChannelMismatch() throws {
        let recipe = StandardMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
            schedulerConfig: .init(),
            backboneConfig: .init(conditioningDim: 4096, latentChannels: 8, maxSequenceLength: 120),
            decoderConfig: .init(inputChannels: 4),
            rendererConfig: ()
        )

        do {
            _ = try DiffusionPipeline(recipe: recipe)
            Issue.record("Expected PipelineError.incompatibleComponents")
        } catch let error as PipelineError {
            switch error {
            case .incompatibleComponents(_, _, let reason):
                #expect(reason.contains("channel") || reason.contains("Channel"))
            default:
                Issue.record("Expected incompatibleComponents, got \(error)")
            }
        }
    }

    // MARK: - Image-to-Image Validation

    @Test("supportsImageToImage with non-BidirectionalDecoder throws")
    func img2imgNonBidirectionalDecoder() throws {
        // MockDecoder does NOT conform to BidirectionalDecoder
        let recipe = StandardMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
            schedulerConfig: .init(),
            backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 120),
            decoderConfig: .init(inputChannels: 4),
            rendererConfig: (),
            supportsImageToImage: true
        )

        do {
            _ = try DiffusionPipeline(recipe: recipe)
            Issue.record("Expected PipelineError.incompatibleComponents for non-BidirectionalDecoder")
        } catch let error as PipelineError {
            switch error {
            case .incompatibleComponents(_, _, let reason):
                #expect(reason.contains("BidirectionalDecoder"))
            default:
                Issue.record("Expected incompatibleComponents, got \(error)")
            }
        }
    }

    @Test("supportsImageToImage with BidirectionalDecoder assembles successfully")
    func img2imgWithBidirectionalDecoder() throws {
        let recipe = BidirectionalMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
            schedulerConfig: .init(),
            backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 120),
            decoderConfig: .init(inputChannels: 4),
            rendererConfig: (),
            supportsImageToImage: true
        )
        let pipeline = try DiffusionPipeline(recipe: recipe)
        #expect(pipeline.memoryRequirement.peakMemoryBytes > 0)
    }
}
