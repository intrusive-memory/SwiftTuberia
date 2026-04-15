@preconcurrency import MLX

// MARK: - Scheduler Supporting Types

public enum BetaSchedule: Sendable {
  case linear(betaStart: Float, betaEnd: Float)
  /// Scaled-linear schedule (HuggingFace `"scaled_linear"`).
  /// betas = linspace(sqrt(betaStart), sqrt(betaEnd), T)²
  /// Used by PixArt-Sigma, Stable Diffusion 1.x/2.x, SDXL.
  case scaledLinear(betaStart: Float, betaEnd: Float)
  case cosine
  case sqrt
}

public enum PredictionType: String, Sendable {
  case epsilon  // predict noise (PixArt, SD, SDXL)
  case velocity  // predict velocity / v-prediction (FLUX)
  case sample  // predict clean sample directly
}

public struct SchedulerPlan: Sendable {
  public let timesteps: [Int]
  public let sigmas: [Float]

  public init(timesteps: [Int], sigmas: [Float]) {
    self.timesteps = timesteps
    self.sigmas = sigmas
  }
}

// MARK: - Scheduler Protocol

public protocol Scheduler: Sendable {
  /// Configuration type for scheduler initialization.
  associatedtype Configuration: Sendable

  /// Construct the scheduler from its configuration. The pipeline calls this
  /// during `DiffusionPipeline.init(recipe:)` to instantiate the component.
  init(configuration: Configuration)

  /// Compute the timestep schedule for a generation run.
  /// - Parameters:
  ///   - steps: Total number of denoising steps.
  ///   - startTimestep: Optional starting timestep for img2img (truncates the plan).
  ///     `nil` = full schedule (text-to-image). Derived from `strength` via
  ///     `Int(Float(steps) * (1.0 - strength))`.
  /// - Returns: A plan containing timesteps and sigmas for the denoising loop.
  func configure(steps: Int, startTimestep: Int?) -> SchedulerPlan

  /// Perform a single denoising step.
  /// - Parameters:
  ///   - output: Model noise prediction from the backbone.
  ///   - timestep: Current timestep.
  ///   - sample: Current noisy latents.
  /// - Returns: Updated (less noisy) latents.
  /// - Throws: `PipelineError` if the step cannot be performed (e.g., scheduler not configured).
  func step(output: MLXArray, timestep: Int, sample: MLXArray) throws -> MLXArray

  /// Add noise to clean latents at a given timestep. Used for img2img initialization.
  func addNoise(to sample: MLXArray, noise: MLXArray, at timestep: Int) -> MLXArray

  /// Clear internal state between generation runs.
  func reset()
}

extension Scheduler {
  public func configure(steps: Int) -> SchedulerPlan {
    configure(steps: steps, startTimestep: nil)
  }
}
