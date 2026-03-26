@preconcurrency import MLX

@testable import Tuberia

/// Mock TextEncoder with configurable shape values for testing.
public final class MockTextEncoder: TextEncoder, @unchecked Sendable {
  public struct Config: Sendable {
    public let embeddingDim: Int
    public let maxSeqLength: Int
    public let componentId: String
    public let estimatedMemory: Int

    public init(
      embeddingDim: Int = 4096,
      maxSeqLength: Int = 120,
      componentId: String = "mock-encoder",
      estimatedMemory: Int = 1_000_000
    ) {
      self.embeddingDim = embeddingDim
      self.maxSeqLength = maxSeqLength
      self.componentId = componentId
      self.estimatedMemory = estimatedMemory
    }
  }

  public typealias Configuration = Config

  private let configuration: Configuration
  private var weights: ModuleParameters?
  public private(set) var isLoaded: Bool = false
  public var encodeCallCount: Int = 0

  public required init(configuration: Configuration) throws {
    self.configuration = configuration
  }

  // MARK: - TextEncoder

  public var outputEmbeddingDim: Int { configuration.embeddingDim }
  public var maxSequenceLength: Int { configuration.maxSeqLength }

  public func encode(_ input: TextEncoderInput) throws -> TextEncoderOutput {
    encodeCallCount += 1
    let batchSize = 1
    let seqLen = min(input.maxLength, configuration.maxSeqLength)
    let embeddings = MLXArray.ones([batchSize, seqLen, configuration.embeddingDim])
    let mask = MLXArray.ones([batchSize, seqLen])
    return TextEncoderOutput(embeddings: embeddings, mask: mask)
  }

  // MARK: - WeightedSegment

  public var estimatedMemoryBytes: Int { configuration.estimatedMemory }

  public var keyMapping: KeyMapping {
    { key in key }
  }

  public func apply(weights: ModuleParameters) throws {
    self.weights = weights
    self.isLoaded = true
  }

  public func unload() {
    self.weights = nil
    self.isLoaded = false
  }
}
