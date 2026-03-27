import Foundation
@preconcurrency import MLX
import Tuberia

/// Flow matching Euler scheduler for FLUX-family models.
///
/// Implements the rectified flow formulation where the forward process is a
/// linear interpolation between data and noise:
///   x_t = (1 - sigma_t) * x_0 + sigma_t * noise
///
/// The Euler step moves along the ODE trajectory defined by the velocity field.
///
/// No model weights -- pure mathematical computation.
/// Conforms to `Scheduler` protocol. Does NOT use BetaSchedule (flow matching
/// uses sigma schedules directly).
public final class FlowMatchEulerScheduler: Scheduler, @unchecked Sendable {
  public typealias Configuration = FlowMatchEulerSchedulerConfiguration

  private let configuration: Configuration
  private var currentPlan: SchedulerPlan?

  public required init(configuration: Configuration) {
    self.configuration = configuration
  }

  // MARK: - Scheduler Protocol

  public func configure(steps: Int, startTimestep: Int?) -> SchedulerPlan {
    let shift = configuration.shift

    // Compute sigma schedule via rectified flow formulation.
    // Sigmas go from ~1.0 (pure noise) down to 0.0 (clean data).
    // With shift, sigmas are computed as: sigma = shift * t / (1 + (shift - 1) * t)
    // where t is linearly spaced from 1.0 to 0.0.
    var sigmas: [Float] = (0...steps).map { i in
      let t = 1.0 - Float(i) / Float(steps)
      if shift == 1.0 {
        return t
      } else {
        // Apply shift transformation
        return shift * t / (1.0 + (shift - 1.0) * t)
      }
    }

    // Truncate for img2img if startTimestep is provided
    if let start = startTimestep, start > 0, start < sigmas.count {
      sigmas = Array(sigmas.dropFirst(start))
    }

    // Timesteps are derived from sigmas (scaled to integer range for compatibility)
    // For flow matching, timesteps map linearly from sigma values
    let timesteps = sigmas.dropLast().enumerated().map { index, sigma in
      Int(sigma * 1000.0)
    }

    let plan = SchedulerPlan(timesteps: Array(timesteps), sigmas: sigmas)
    self.currentPlan = plan
    return plan
  }

  public func step(output: MLXArray, timestep: Int, sample: MLXArray) throws -> MLXArray {
    guard let plan = currentPlan else {
      throw PipelineError.generationFailed(
        step: timestep,
        reason: "Scheduler not configured. Call configure(steps:) before step()."
      )
    }

    // Find the current index in the plan
    guard let currentIdx = plan.timesteps.firstIndex(of: timestep) else {
      // Snap to nearest timestep instead of silently dropping the step
      let nearestIdx = findNearestTimestepIndex(for: timestep, in: plan.timesteps)
      let sigmaI = plan.sigmas[nearestIdx]
      let sigmaNext = plan.sigmas[nearestIdx + 1]
      let dt = sigmaNext - sigmaI

      return sample + Float(dt) * output
    }

    // Euler step: x_{t+dt} = x_t + dt * v(x_t, t)
    // where dt = sigma_{i+1} - sigma_i (negative, since sigma decreases)
    let sigmaI = plan.sigmas[currentIdx]
    let sigmaNext = plan.sigmas[currentIdx + 1]
    let dt = sigmaNext - sigmaI

    return sample + Float(dt) * output
  }

  /// Find the nearest timestep index in the schedule to the given timestep.
  private func findNearestTimestepIndex(for timestep: Int, in timesteps: [Int]) -> Int {
    guard !timesteps.isEmpty else { return 0 }

    var minDistance = Int.max
    var nearestIdx = 0

    for (idx, t) in timesteps.enumerated() {
      let distance = abs(t - timestep)
      if distance < minDistance {
        minDistance = distance
        nearestIdx = idx
      }
    }

    // Ensure we don't go out of bounds for sigma access
    return min(nearestIdx, timesteps.count - 2)
  }

  public func addNoise(to sample: MLXArray, noise: MLXArray, at timestep: Int) -> MLXArray {
    // For flow matching: x_t = (1 - sigma) * x_0 + sigma * noise
    let sigma = Float(timestep) / 1000.0
    return (1.0 - sigma) * sample + sigma * noise
  }

  public func reset() {
    currentPlan = nil
  }
}
