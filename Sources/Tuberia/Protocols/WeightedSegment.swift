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
}

extension WeightedSegment {
  public var tensorTransform: TensorTransform? { nil }
}
