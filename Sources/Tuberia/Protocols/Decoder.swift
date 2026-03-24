@preconcurrency import MLX

// MARK: - Decoder Metadata

public protocol DecoderMetadata: Sendable {
    var scalingFactor: Float { get }
}

public struct ImageDecoderMetadata: DecoderMetadata, Sendable {
    public let scalingFactor: Float
    public init(scalingFactor: Float) { self.scalingFactor = scalingFactor }
}

public struct AudioDecoderMetadata: DecoderMetadata, Sendable {
    public let scalingFactor: Float
    public let sampleRate: Int
    public init(scalingFactor: Float, sampleRate: Int) {
        self.scalingFactor = scalingFactor
        self.sampleRate = sampleRate
    }
}

// MARK: - Decoded Output

public struct DecodedOutput: Sendable {
    /// Decoded data. Shape: [B, H, W, C] for images, [B, samples] for audio.
    public let data: MLXArray
    /// Modality-specific metadata for the Renderer.
    public let metadata: any DecoderMetadata

    public init(data: MLXArray, metadata: any DecoderMetadata) {
        self.data = data
        self.metadata = metadata
    }
}

// MARK: - Decoder Protocol

public protocol Decoder: WeightedSegment {
    /// Configuration type -- must include the Acervo component ID for weights.
    associatedtype Configuration: Sendable

    /// Construct the decoder from its configuration. The pipeline calls this
    /// during `DiffusionPipeline.init(recipe:)` to instantiate the component.
    init(configuration: Configuration) throws

    /// Expected number of latent channels from the connected Backbone.
    /// Validated at assembly: must equal `Backbone.outputLatentChannels`.
    var expectedInputChannels: Int { get }

    /// VAE latent scaling factor. Applied internally by the decoder
    /// (`latents * (1.0 / scalingFactor)`) -- the pipeline does NOT touch this.
    var scalingFactor: Float { get }

    /// Decode latents into output data.
    func decode(_ latents: MLXArray) throws -> DecodedOutput
}

/// A Decoder that can also encode (pixels -> latents).
/// Required for image-to-image and inpainting generation modes.
public protocol BidirectionalDecoder: Decoder {
    /// Encode pixel data into the latent space.
    /// Input:  [B, H, W, 3] (normalized float pixels)
    /// Output: [B, H/f, W/f, C] (latents)
    func encode(_ pixels: MLXArray) throws -> MLXArray
}
