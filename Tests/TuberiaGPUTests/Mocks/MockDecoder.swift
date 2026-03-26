@preconcurrency import MLX

@testable import Tuberia

/// Mock Decoder with configurable shape values for testing.
/// Can optionally conform to BidirectionalDecoder for img2img tests.
public final class MockDecoder: Decoder, @unchecked Sendable {
  public struct Config: Sendable {
    public let inputChannels: Int
    public let scalingFactor: Float
    public let componentId: String
    public let estimatedMemory: Int

    public init(
      inputChannels: Int = 4,
      scalingFactor: Float = 0.13025,
      componentId: String = "mock-decoder",
      estimatedMemory: Int = 500_000
    ) {
      self.inputChannels = inputChannels
      self.scalingFactor = scalingFactor
      self.componentId = componentId
      self.estimatedMemory = estimatedMemory
    }
  }

  public typealias Configuration = Config

  private let configuration: Configuration
  private var weights: ModuleParameters?
  public private(set) var isLoaded: Bool = false
  public var decodeCallCount: Int = 0

  public required init(configuration: Configuration) throws {
    self.configuration = configuration
  }

  // MARK: - Decoder

  public var expectedInputChannels: Int { configuration.inputChannels }
  public var scalingFactor: Float { configuration.scalingFactor }

  public func decode(_ latents: MLXArray) throws -> DecodedOutput {
    decodeCallCount += 1
    // Simulate VAE decode: latents [B, H, W, C] -> pixels [B, H*8, W*8, 3]
    let shape = latents.shape
    let batchSize = shape[0]
    let height = shape.count > 1 ? shape[1] * 8 : 64
    let width = shape.count > 2 ? shape[2] * 8 : 64
    let pixelData = MLXArray.ones([batchSize, height, width, 3]) * 0.5
    let metadata = ImageDecoderMetadata(scalingFactor: configuration.scalingFactor)
    return DecodedOutput(data: pixelData, metadata: metadata)
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

/// Mock BidirectionalDecoder for img2img testing.
public final class MockBidirectionalDecoder: BidirectionalDecoder, @unchecked Sendable {
  public struct Config: Sendable {
    public let inputChannels: Int
    public let scalingFactor: Float
    public let componentId: String
    public let estimatedMemory: Int

    public init(
      inputChannels: Int = 4,
      scalingFactor: Float = 0.13025,
      componentId: String = "mock-bidirectional-decoder",
      estimatedMemory: Int = 500_000
    ) {
      self.inputChannels = inputChannels
      self.scalingFactor = scalingFactor
      self.componentId = componentId
      self.estimatedMemory = estimatedMemory
    }
  }

  public typealias Configuration = Config

  private let configuration: Configuration
  private var weights: ModuleParameters?
  public private(set) var isLoaded: Bool = false
  public var decodeCallCount: Int = 0
  public var encodeCallCount: Int = 0

  public required init(configuration: Configuration) throws {
    self.configuration = configuration
  }

  // MARK: - Decoder

  public var expectedInputChannels: Int { configuration.inputChannels }
  public var scalingFactor: Float { configuration.scalingFactor }

  public func decode(_ latents: MLXArray) throws -> DecodedOutput {
    decodeCallCount += 1
    let shape = latents.shape
    let batchSize = shape[0]
    let height = shape.count > 1 ? shape[1] * 8 : 64
    let width = shape.count > 2 ? shape[2] * 8 : 64
    let pixelData = MLXArray.ones([batchSize, height, width, 3]) * 0.5
    let metadata = ImageDecoderMetadata(scalingFactor: configuration.scalingFactor)
    return DecodedOutput(data: pixelData, metadata: metadata)
  }

  // MARK: - BidirectionalDecoder

  public func encode(_ pixels: MLXArray) throws -> MLXArray {
    encodeCallCount += 1
    // Simulate VAE encode: pixels [B, H, W, 3] -> latents [B, H/8, W/8, C]
    let shape = pixels.shape
    let batchSize = shape[0]
    let height = shape.count > 1 ? shape[1] / 8 : 8
    let width = shape.count > 2 ? shape[2] / 8 : 8
    return MLXArray.ones([batchSize, height, width, configuration.inputChannels]) * 0.5
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
