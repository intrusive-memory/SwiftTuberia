import Foundation

/// Result of a diffusion generation run.
public struct DiffusionGenerationResult: Sendable {
  public let output: RenderedOutput
  public let seed: UInt32
  public let steps: Int
  public let guidanceScale: Float
  public let duration: TimeInterval

  public init(
    output: RenderedOutput, seed: UInt32, steps: Int,
    guidanceScale: Float, duration: TimeInterval
  ) {
    self.output = output
    self.seed = seed
    self.steps = steps
    self.guidanceScale = guidanceScale
    self.duration = duration
  }
}
