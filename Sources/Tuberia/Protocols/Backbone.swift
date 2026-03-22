@preconcurrency import MLX

// MARK: - Backbone Input

public struct BackboneInput: Sendable {
    /// Noisy latents. Shape: [B, spatial..., channels]
    public let latents: MLXArray
    /// Text encoder embeddings (mapped from TextEncoderOutput.embeddings).
    /// Shape: [B, seq, dim]
    public let conditioning: MLXArray
    /// Text encoder mask (mapped from TextEncoderOutput.mask).
    /// Shape: [B, seq]
    public let conditioningMask: MLXArray
    /// Current denoising timestep. Scalar or [B].
    public let timestep: MLXArray

    public init(latents: MLXArray, conditioning: MLXArray,
                conditioningMask: MLXArray, timestep: MLXArray) {
        self.latents = latents
        self.conditioning = conditioning
        self.conditioningMask = conditioningMask
        self.timestep = timestep
    }
}

// MARK: - Backbone Protocol

public protocol Backbone: WeightedSegment {
    /// Configuration type -- must include the Acervo component ID for weights.
    associatedtype Configuration: Sendable

    /// Construct the backbone from its configuration. The pipeline calls this
    /// during `DiffusionPipeline.init(recipe:)` to instantiate the component.
    init(configuration: Configuration) throws

    /// Expected embedding dimension from the connected TextEncoder.
    /// Validated at assembly: must equal `TextEncoder.outputEmbeddingDim`.
    var expectedConditioningDim: Int { get }

    /// Number of latent channels produced by the backbone.
    /// Validated at assembly: must equal `Decoder.expectedInputChannels`.
    var outputLatentChannels: Int { get }

    /// Maximum sequence length the backbone expects from the TextEncoder.
    /// Validated at assembly: must equal the TextEncoder configuration's `maxSequenceLength`.
    /// This ensures the encoder won't produce sequences longer than the backbone can attend to.
    var expectedMaxSequenceLength: Int { get }

    /// Forward pass -- noise prediction.
    /// - Parameter input: Conditioned latents and timestep.
    /// - Returns: Noise prediction. Shape: [B, spatial..., channels]
    func forward(_ input: BackboneInput) throws -> MLXArray
}
