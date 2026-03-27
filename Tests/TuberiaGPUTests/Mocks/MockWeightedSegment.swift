@preconcurrency import MLX

@testable import Tuberia

/// Standalone mock WeightedSegment for testing the WeightedSegment lifecycle
/// independent of any specific pipe segment protocol.
public final class MockWeightedSegment: WeightedSegment, @unchecked Sendable {
  private var weights: ModuleParameters?
  public private(set) var isLoaded: Bool = false
  public let estimatedMemoryBytes: Int
  public var requiredKeys: Set<String>?
  public var applyCallCount: Int = 0
  public var unloadCallCount: Int = 0

  /// - Parameters:
  ///   - estimatedMemoryBytes: Memory estimate for this segment.
  ///   - requiredKeys: If non-nil, `apply(weights:)` will throw if any required key is missing.
  public init(estimatedMemoryBytes: Int = 100_000, requiredKeys: Set<String>? = nil) {
    self.estimatedMemoryBytes = estimatedMemoryBytes
    self.requiredKeys = requiredKeys
  }

  public var keyMapping: KeyMapping {
    { key in key }
  }

  public func apply(weights: ModuleParameters) throws {
    applyCallCount += 1
    if let required = requiredKeys {
      let missing = required.subtracting(weights.parameters.keys)
      if !missing.isEmpty {
        throw PipelineError.weightLoadingFailed(
          component: "MockWeightedSegment",
          reason: "Missing required keys: \(missing.sorted().joined(separator: ", "))"
        )
      }
    }
    self.weights = weights
    self.isLoaded = true
  }

  public func unload() {
    unloadCallCount += 1
    self.weights = nil
    self.isLoaded = false
  }

  /// Access to stored weights for test assertions.
  public var storedWeights: ModuleParameters? { weights }

  public var currentWeights: ModuleParameters? { weights }
}
