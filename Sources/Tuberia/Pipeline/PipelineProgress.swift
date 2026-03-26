import Foundation

/// Unified progress reporting for all pipeline operations.
public enum PipelineProgress: Sendable {
  case downloading(component: String, fraction: Double)
  case loading(component: String, fraction: Double)
  case encoding(fraction: Double)
  case generating(step: Int, totalSteps: Int, elapsed: TimeInterval)
  case decoding
  case rendering
  case complete(duration: TimeInterval)
}
