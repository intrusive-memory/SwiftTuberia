import Foundation
@preconcurrency import MLX
import Tuberia

/// DPM-Solver++ multistep scheduler for diffusion denoising.
///
/// Implements the DPM-Solver++ algorithm (Lu et al., 2022) with support for
/// first-order (Euler) and second-order (midpoint) methods.
///
/// No model weights -- pure mathematical computation.
/// Conforms to `Scheduler` protocol. Stateful within a generation (tracks
/// previous step outputs for multistep), stateless between generations (via `reset()`).
public final class DPMSolverScheduler: Scheduler, @unchecked Sendable {
  public typealias Configuration = DPMSolverSchedulerConfiguration

  private let configuration: Configuration

  // Precomputed noise schedule values
  private let alphasCumprod: [Float]
  private let trainTimesteps: Int

  // State for multistep (second-order)
  private var previousOutputs: [MLXArray] = []
  private var currentPlan: SchedulerPlan?

  public required init(configuration: Configuration) {
    self.configuration = configuration
    self.trainTimesteps = configuration.trainTimesteps

    // Compute betas and cumulative alphas
    let betas = Self.computeBetas(
      schedule: configuration.betaSchedule,
      trainTimesteps: configuration.trainTimesteps
    )

    // alphas = 1 - betas
    let alphas = betas.map { 1.0 - $0 }

    // alphas_cumprod = cumulative product of alphas
    var cumprod: [Float] = []
    var product: Float = 1.0
    for alpha in alphas {
      product *= alpha
      cumprod.append(product)
    }
    self.alphasCumprod = cumprod
  }

  // MARK: - Scheduler Protocol

  public func configure(steps: Int, startTimestep: Int?) -> SchedulerPlan {
    // Compute evenly spaced timesteps from trainTimesteps-1 down to 0
    let stepRatio = Float(trainTimesteps) / Float(steps)
    var timesteps: [Int] = (0..<steps).map { i in
      Int(Float(trainTimesteps - 1) - Float(i) * stepRatio + 0.5)
    }

    // Ensure we don't have negative timesteps
    timesteps = timesteps.map { max(0, $0) }

    // Truncate for img2img if startTimestep is provided
    if let start = startTimestep, start > 0, start < timesteps.count {
      timesteps = Array(timesteps.dropFirst(start))
    }

    // Compute sigmas (sqrt((1 - alpha_cumprod) / alpha_cumprod)) for each timestep
    let sigmas = timesteps.map { t -> Float in
      let idx = min(t, alphasCumprod.count - 1)
      let alphaCumprod = alphasCumprod[max(0, idx)]
      return sqrt((1.0 - alphaCumprod) / max(alphaCumprod, 1e-8))
    }

    let plan = SchedulerPlan(timesteps: timesteps, sigmas: sigmas)
    self.currentPlan = plan
    return plan
  }

  public func step(output: MLXArray, timestep: Int, sample: MLXArray) throws -> MLXArray {
    let idx = min(timestep, alphasCumprod.count - 1)
    let alphaProdT = alphasCumprod[max(0, idx)]

    // Convert model output to predicted x0 based on prediction type
    let predictedOriginal: MLXArray
    switch configuration.predictionType {
    case .epsilon:
      // x0 = (x_t - sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)
      let sqrtAlpha = sqrt(alphaProdT)
      let sqrtOneMinusAlpha = sqrt(1.0 - alphaProdT)
      predictedOriginal = (sample - Float(sqrtOneMinusAlpha) * output) / Float(sqrtAlpha)

    case .velocity:
      // x0 = sqrt(alpha_t) * x_t - sqrt(1 - alpha_t) * v
      let sqrtAlpha = sqrt(alphaProdT)
      let sqrtOneMinusAlpha = sqrt(1.0 - alphaProdT)
      predictedOriginal = Float(sqrtAlpha) * sample - Float(sqrtOneMinusAlpha) * output

    case .sample:
      // Model directly predicts x0
      predictedOriginal = output
    }

    // DPM-Solver++ update step
    let result: MLXArray

    if configuration.solverOrder == 1 || previousOutputs.isEmpty {
      // First-order (Euler-like) step
      result = dpmSolverFirstOrderStep(
        predictedOriginal: predictedOriginal,
        timestep: timestep,
        sample: sample
      )
    } else {
      // Second-order (midpoint) step using previous output
      let prevPredicted = previousOutputs.last!
      result = dpmSolverSecondOrderStep(
        predictedOriginal: predictedOriginal,
        previousPredicted: prevPredicted,
        timestep: timestep,
        sample: sample
      )
    }

    // Store for multistep
    previousOutputs.append(predictedOriginal)
    // Keep only the last (solverOrder - 1) outputs for multistep
    if previousOutputs.count > configuration.solverOrder {
      previousOutputs.removeFirst()
    }

    return result
  }

  public func addNoise(to sample: MLXArray, noise: MLXArray, at timestep: Int) -> MLXArray {
    let idx = min(timestep, alphasCumprod.count - 1)
    let alphaCumprod = alphasCumprod[max(0, idx)]
    let sqrtAlpha = Float(sqrt(alphaCumprod))
    let sqrtOneMinusAlpha = Float(sqrt(1.0 - alphaCumprod))

    return sqrtAlpha * sample + sqrtOneMinusAlpha * noise
  }

  public func reset() {
    previousOutputs.removeAll()
    currentPlan = nil
  }

  // MARK: - DPM-Solver++ Steps

  /// First-order DPM-Solver++ step (equivalent to DDIM).
  private func dpmSolverFirstOrderStep(
    predictedOriginal: MLXArray,
    timestep: Int,
    sample: MLXArray
  ) -> MLXArray {
    // Compute the previous timestep's alpha
    let idx = min(timestep, alphasCumprod.count - 1)
    let alphaProdT = alphasCumprod[max(0, idx)]

    // Find the next timestep in the plan
    let prevTimestep = findPreviousTimestep(current: timestep)
    let alphaProdPrev: Float
    if prevTimestep >= 0 && prevTimestep < alphasCumprod.count {
      alphaProdPrev = alphasCumprod[prevTimestep]
    } else {
      alphaProdPrev = 1.0  // Final step: fully denoised
    }

    // DPM-Solver++ first-order:
    // x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + sqrt(1 - alpha_{t-1}) * eps_pred
    // where eps_pred = (x_t - sqrt(alpha_t) * x0_pred) / sqrt(1 - alpha_t)
    let sqrtAlphaPrev = Float(sqrt(alphaProdPrev))
    let sqrtOneMinusAlphaPrev = Float(sqrt(1.0 - alphaProdPrev))
    let sqrtAlphaT = Float(sqrt(alphaProdT))
    let sqrtOneMinusAlphaT = Float(sqrt(max(1.0 - alphaProdT, 1e-8)))

    // Predicted noise from current timestep
    let predictedNoise = (sample - sqrtAlphaT * predictedOriginal) / sqrtOneMinusAlphaT

    return sqrtAlphaPrev * predictedOriginal + sqrtOneMinusAlphaPrev * predictedNoise
  }

  /// Second-order DPM-Solver++ step (midpoint method).
  private func dpmSolverSecondOrderStep(
    predictedOriginal: MLXArray,
    previousPredicted: MLXArray,
    timestep: Int,
    sample: MLXArray
  ) -> MLXArray {
    // For second-order, we use a linear combination of current and previous predictions
    // This implements the DPM-Solver++(2M) update:
    // x0_corrected = 0.5 * (3 * x0_current - x0_previous)
    let correctedPrediction = 1.5 * predictedOriginal - 0.5 * previousPredicted

    // Then apply the first-order step with the corrected prediction
    return dpmSolverFirstOrderStep(
      predictedOriginal: correctedPrediction,
      timestep: timestep,
      sample: sample
    )
  }

  // MARK: - Beta Schedule Computation

  private static func computeBetas(schedule: BetaSchedule, trainTimesteps: Int) -> [Float] {
    switch schedule {
    case .linear(let betaStart, let betaEnd):
      // Linear interpolation from betaStart to betaEnd
      return (0..<trainTimesteps).map { i in
        let t = Float(i) / Float(trainTimesteps - 1)
        return betaStart + t * (betaEnd - betaStart)
      }

    case .cosine:
      // Cosine schedule (Nichol & Dhariwal, 2021)
      let maxBeta: Float = 0.999
      return (0..<trainTimesteps).map { i in
        let t1 = Float(i) / Float(trainTimesteps)
        let t2 = Float(i + 1) / Float(trainTimesteps)
        let alpha1 = cos((t1 + 0.008) / 1.008 * Float.pi / 2.0)
        let alpha2 = cos((t2 + 0.008) / 1.008 * Float.pi / 2.0)
        return min(1.0 - (alpha2 * alpha2) / (alpha1 * alpha1), maxBeta)
      }

    case .sqrt:
      // Square root schedule
      return (0..<trainTimesteps).map { i in
        let t = Float(i) / Float(trainTimesteps - 1)
        return sqrt(t) * 0.02
      }
    }
  }

  // MARK: - Helpers

  /// Find the previous timestep in the current plan.
  private func findPreviousTimestep(current: Int) -> Int {
    guard let plan = currentPlan else {
      return max(current - 1, 0)
    }

    if let idx = plan.timesteps.firstIndex(of: current), idx + 1 < plan.timesteps.count {
      return plan.timesteps[idx + 1]
    }

    // If we're at the last timestep, return 0 (fully denoised)
    return 0
  }
}
