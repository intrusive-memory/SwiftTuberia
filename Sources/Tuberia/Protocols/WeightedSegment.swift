/// Lifecycle for pipe segments that carry model weights.
/// Conformers MUST be `final class` with `@unchecked Sendable`.
/// The DiffusionPipeline actor serializes all access.
public protocol WeightedSegment: Sendable {
  func apply(weights: ModuleParameters) throws
  func unload()
  var estimatedMemoryBytes: Int { get }
  var isLoaded: Bool { get }
  var keyMapping: KeyMapping { get }
  var tensorTransform: TensorTransform? { get }
  /// The last `ModuleParameters` applied via `apply(weights:)`, or `nil` if unloaded.
  /// Used by LoRA integration to merge/restore adapter weights into the base model.
  var currentWeights: ModuleParameters? { get }
}

extension WeightedSegment {
  public var tensorTransform: TensorTransform? { nil }
}
