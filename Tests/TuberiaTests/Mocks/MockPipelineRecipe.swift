@testable import Tuberia

/// Configurable mock PipelineRecipe for testing pipeline assembly and validation.
public struct MockPipelineRecipe<
    E: TextEncoder,
    S: Scheduler,
    B: Backbone,
    D: Decoder,
    R: Renderer
>: PipelineRecipe, Sendable
    where E.Configuration: Sendable, S.Configuration: Sendable,
          B.Configuration: Sendable, D.Configuration: Sendable,
          R.Configuration: Sendable
{
    public typealias Encoder = E
    public typealias Sched = S
    public typealias Back = B
    public typealias Dec = D
    public typealias Rend = R

    public let encoderConfig: E.Configuration
    public let schedulerConfig: S.Configuration
    public let backboneConfig: B.Configuration
    public let decoderConfig: D.Configuration
    public let rendererConfig: R.Configuration

    public let supportsImageToImage: Bool
    public let unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy
    public let allComponentIds: [String]

    private let _quantizationConfig: QuantizationConfig
    private let _customValidation: (@Sendable () throws -> Void)?

    public init(
        encoderConfig: E.Configuration,
        schedulerConfig: S.Configuration,
        backboneConfig: B.Configuration,
        decoderConfig: D.Configuration,
        rendererConfig: R.Configuration,
        supportsImageToImage: Bool = false,
        unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .emptyPrompt,
        allComponentIds: [String] = [],
        quantizationConfig: QuantizationConfig = .asStored,
        customValidation: (@Sendable () throws -> Void)? = nil
    ) {
        self.encoderConfig = encoderConfig
        self.schedulerConfig = schedulerConfig
        self.backboneConfig = backboneConfig
        self.decoderConfig = decoderConfig
        self.rendererConfig = rendererConfig
        self.supportsImageToImage = supportsImageToImage
        self.unconditionalEmbeddingStrategy = unconditionalEmbeddingStrategy
        self.allComponentIds = allComponentIds
        self._quantizationConfig = quantizationConfig
        self._customValidation = customValidation
    }

    public func quantizationFor(_ role: PipelineRole) -> QuantizationConfig {
        _quantizationConfig
    }

    public func validate() throws {
        try _customValidation?()
    }
}

/// Convenience typealias for the most common mock recipe.
public typealias StandardMockRecipe = MockPipelineRecipe<
    MockTextEncoder,
    MockScheduler,
    MockBackbone,
    MockDecoder,
    MockRenderer
>

/// Convenience typealias for mock recipe with BidirectionalDecoder.
public typealias BidirectionalMockRecipe = MockPipelineRecipe<
    MockTextEncoder,
    MockScheduler,
    MockBackbone,
    MockBidirectionalDecoder,
    MockRenderer
>
