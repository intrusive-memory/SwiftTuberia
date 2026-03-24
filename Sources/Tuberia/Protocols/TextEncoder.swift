@preconcurrency import MLX

// MARK: - TextEncoder Input/Output

public struct TextEncoderInput: Sendable {
    public let text: String
    public let maxLength: Int

    public init(text: String, maxLength: Int) {
        self.text = text
        self.maxLength = maxLength
    }
}

public struct TextEncoderOutput: Sendable {
    /// Dense embeddings. Shape: [B, seq, dim]
    public let embeddings: MLXArray
    /// Attention mask. Shape: [B, seq]. 1 = real token, 0 = padding.
    public let mask: MLXArray

    public init(embeddings: MLXArray, mask: MLXArray) {
        self.embeddings = embeddings
        self.mask = mask
    }
}

// MARK: - TextEncoder Protocol

public protocol TextEncoder: WeightedSegment {
    /// Configuration type -- must include the Acervo component ID for weights
    /// and any non-weight resources (e.g., tokenizer component ID).
    associatedtype Configuration: Sendable

    /// Construct the encoder from its configuration. The pipeline calls this
    /// during `DiffusionPipeline.init(recipe:)` to instantiate the component.
    init(configuration: Configuration) throws

    /// Embedding dimension of the encoder output. Used for assembly-time validation
    /// against `Backbone.expectedConditioningDim`.
    var outputEmbeddingDim: Int { get }

    /// Maximum sequence length this encoder produces. Used for assembly-time validation
    /// against `Backbone.expectedMaxSequenceLength`.
    var maxSequenceLength: Int { get }

    /// Encode text into dense embeddings.
    func encode(_ input: TextEncoderInput) throws -> TextEncoderOutput
}
