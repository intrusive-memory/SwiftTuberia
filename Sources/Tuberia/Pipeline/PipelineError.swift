/// Errors scoped to pipeline assembly, infrastructure, generation, and cancellation.
public enum PipelineError: Error {
  // Assembly errors (caught before generation)
  case incompatibleComponents(inlet: String, outlet: String, reason: String)
  case missingComponent(role: String)

  // Infrastructure errors
  case modelNotDownloaded(component: String)
  case insufficientMemory(required: UInt64, available: UInt64, component: String)
  case weightLoadingFailed(component: String, reason: String)
  case downloadFailed(component: String, reason: String)

  // Generation errors
  case encodingFailed(reason: String)
  case generationFailed(step: Int, reason: String)
  case decodingFailed(reason: String)
  case renderingFailed(reason: String)

  // Cancellation
  case cancelled
}
