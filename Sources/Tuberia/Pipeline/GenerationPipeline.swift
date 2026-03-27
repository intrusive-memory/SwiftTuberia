/// Memory requirements for a pipeline, supporting both eager and phased loading.
public struct MemoryRequirement: Sendable {
  /// Total memory if all components loaded simultaneously.
  public let peakMemoryBytes: UInt64
  /// Maximum memory needed for any single loading phase.
  public let phasedMemoryBytes: UInt64

  public init(peakMemoryBytes: UInt64, phasedMemoryBytes: UInt64) {
    self.peakMemoryBytes = peakMemoryBytes
    self.phasedMemoryBytes = phasedMemoryBytes
  }
}

/// Top-level protocol that any generation pipeline conforms to.
///
/// A pipeline manages the lifecycle of all its components and orchestrates
/// the generation flow. Pipelines are actors -- one generation at a time per instance.
public protocol GenerationPipeline: Sendable {
  associatedtype Request: Sendable
  associatedtype Result: Sendable

  func generate(request: Request, progress: @Sendable (PipelineProgress) -> Void) async throws
    -> Result
  func loadModels(progress: @Sendable (Double, String) -> Void) async throws
  func unloadModels() async
  var memoryRequirement: MemoryRequirement { get }
  var isLoaded: Bool { get }
}
