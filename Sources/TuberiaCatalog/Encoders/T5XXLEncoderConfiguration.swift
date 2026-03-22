import Tuberia

// MARK: - T5XXLEncoder Configuration

public struct T5XXLEncoderConfiguration: Sendable {
    /// Acervo component ID for weights AND tokenizer files (bundled together).
    public let componentId: String
    /// Maximum sequence length for tokenization.
    public let maxSequenceLength: Int
    /// Embedding dimension (informational -- always 4096 for T5-XXL).
    public let embeddingDim: Int

    public init(
        componentId: String = "t5-xxl-encoder-int4",
        maxSequenceLength: Int = 120,
        embeddingDim: Int = 4096
    ) {
        self.componentId = componentId
        self.maxSequenceLength = maxSequenceLength
        self.embeddingDim = embeddingDim
    }
}
