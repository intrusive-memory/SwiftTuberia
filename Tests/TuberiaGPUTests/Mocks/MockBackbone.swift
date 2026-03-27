@preconcurrency import MLX

@testable import Tuberia

/// Mock Backbone with configurable shape values for testing.
public final class MockBackbone: Backbone, @unchecked Sendable {
  public struct Config: Sendable {
    public let conditioningDim: Int
    public let latentChannels: Int
    public let maxSequenceLength: Int
    public let componentId: String
    public let estimatedMemory: Int

    public init(
      conditioningDim: Int = 4096,
      latentChannels: Int = 4,
      maxSequenceLength: Int = 120,
      componentId: String = "mock-backbone",
      estimatedMemory: Int = 2_000_000
    ) {
      self.conditioningDim = conditioningDim
      self.latentChannels = latentChannels
      self.maxSequenceLength = maxSequenceLength
      self.componentId = componentId
      self.estimatedMemory = estimatedMemory
    }
  }

  public typealias Configuration = Config

  private let configuration: Configuration
  private var weights: ModuleParameters?
  public private(set) var isLoaded: Bool = false
  public var forwardCallCount: Int = 0

  public required init(configuration: Configuration) throws {
    self.configuration = configuration
  }

  // MARK: - Backbone

  public var expectedConditioningDim: Int { configuration.conditioningDim }
  public var outputLatentChannels: Int { configuration.latentChannels }
  public var expectedMaxSequenceLength: Int { configuration.maxSequenceLength }

  public func forward(_ input: BackboneInput) throws -> MLXArray {
    forwardCallCount += 1
    // Return a tensor with the same shape as input latents (noise prediction)
    return MLXArray.ones(input.latents.shape)
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

  public var currentWeights: ModuleParameters? { weights }

  public func unload() {
    self.weights = nil
    self.isLoaded = false
  }
}
