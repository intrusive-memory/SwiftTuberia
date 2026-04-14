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
  // Stores (predicted_x0, timestep) pairs from previous steps.
  private var previousOutputs: [(MLXArray, Int)] = []
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
      // Second-order (midpoint) step using previous x0 prediction + its timestep
      let (prevPredicted, prevTimestep) = previousOutputs.last!
      result = dpmSolverSecondOrderStep(
        predictedOriginal: predictedOriginal,
        previousPredicted: prevPredicted,
        timestep: timestep,
        previousTimestep: prevTimestep,
        sample: sample
      )
    }

    // Store x0 prediction + current timestep for next step's second-order correction.
    previousOutputs.append((predictedOriginal, timestep))
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
    previousOutputs.removeAll(keepingCapacity: true)
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

  /// Second-order DPM-Solver++(2M) step (midpoint method).
  ///
  /// Implements the exact DPM-Solver++(2M) update from Lu et al. (2022), matching
  /// the diffusers `DPMSolverMultistepScheduler` with `solver_type="midpoint"`:
  ///
  /// ```
  /// lambda_t  = log(alpha_t)  - log(sigma_t)   (log-SNR at target timestep)
  /// lambda_s0 = log(alpha_s0) - log(sigma_s0)  (log-SNR at current timestep)
  /// lambda_s1 = log(alpha_s1) - log(sigma_s1)  (log-SNR at previous timestep)
  /// h   = lambda_t  - lambda_s0
  /// h_0 = lambda_s0 - lambda_s1
  /// r0  = h_0 / h
  /// D0  = m0                         (current denoised prediction)
  /// D1  = (1/r0) * (m0 - m1)         (first-order finite difference)
  /// x_t = (sigma_t/sigma_s0)*sample - alpha_t*(exp(-h)-1)*D0 - 0.5*alpha_t*(exp(-h)-1)*D1
  /// ```
  ///
  /// where:
  /// - `alpha_t   = sqrt(alphaCumprod[t_target])`
  /// - `sigma_t   = sqrt(1 - alphaCumprod[t_target])`
  /// - `alpha_s0  = sqrt(alphaCumprod[t_current])`
  /// - `sigma_s0  = sqrt(1 - alphaCumprod[t_current])`
  /// - `alpha_s1  = sqrt(alphaCumprod[t_previous])`
  /// - `sigma_s1  = sqrt(1 - alphaCumprod[t_previous])`
  private func dpmSolverSecondOrderStep(
    predictedOriginal: MLXArray,   // m0: current denoised prediction (x0 at s0)
    previousPredicted: MLXArray,   // m1: previous denoised prediction (x0 at s1)
    timestep: Int,                 // s0: current timestep
    previousTimestep: Int,         // s1: previous timestep (one step further back)
    sample: MLXArray               // x_{s0}: current noisy sample
  ) -> MLXArray {
    // Target timestep = the one we're stepping TO (next in the denoising direction)
    let targetTimestep = findPreviousTimestep(current: timestep)  // t (lower noise)

    // Alpha cumulative products for t, s0, s1
    let acT = alphaCumprod(at: targetTimestep)
    let acS0 = alphaCumprod(at: timestep)
    let acS1 = alphaCumprod(at: previousTimestep)

    // alpha and sigma in DPM-Solver++ notation (alpha = sqrt(ac), sigma = sqrt(1-ac))
    let alphaT  = Float(Foundation.sqrt(acT))
    let sigmaT  = Float(Foundation.sqrt(max(1.0 - acT, 1e-8)))
    let alphaS0 = Float(Foundation.sqrt(acS0))
    let sigmaS0 = Float(Foundation.sqrt(max(1.0 - acS0, 1e-8)))
    let alphaS1 = Float(Foundation.sqrt(acS1))
    let sigmaS1 = Float(Foundation.sqrt(max(1.0 - acS1, 1e-8)))

    // Log-SNR: lambda = log(alpha) - log(sigma)
    let lambdaT  = Foundation.log(alphaT)  - Foundation.log(sigmaT)
    let lambdaS0 = Foundation.log(alphaS0) - Foundation.log(sigmaS0)
    let lambdaS1 = Foundation.log(alphaS1) - Foundation.log(sigmaS1)

    // Log-SNR differences
    let h  = lambdaT  - lambdaS0  // h > 0 means moving toward lower noise
    let h0 = lambdaS0 - lambdaS1  // h0 > 0 (s1 is earlier / higher noise than s0)
    let r0 = max(h0 / h, 1e-8)    // ratio, guard against zero

    // D0 = current prediction, D1 = first-order finite difference
    // D1 = (m0 - m1) / r0
    // Compute as MLX: D1_coefficient = Float(1.0/r0)
    let d1Coeff = Float(1.0 / r0)
    let d1 = d1Coeff * (predictedOriginal - previousPredicted)

    // DPM-Solver++(2M) midpoint update:
    // x_t = (sigma_t/sigma_s0)*sample - alpha_t*(exp(-h)-1)*(D0 + 0.5*D1)
    let expMinusH = Foundation.exp(-h)
    let coeff = Float(-alphaT * (expMinusH - 1.0))  // positive when exp(-h) < 1 (h > 0)
    let sigmaTOverS0 = Float(sigmaT / sigmaS0)

    return sigmaTOverS0 * sample + coeff * (predictedOriginal + 0.5 * d1)
  }

  /// Look up alpha_cumprod at the given timestep index (clamped to valid range).
  ///
  /// Returns 1.0 for the fully-denoised terminal state (timestep < 0), which
  /// is only used when `findPreviousTimestep` returns -1 (not applicable here).
  /// For t=0, returns alphasCumprod[0] ≈ 0.9999 (effectively denoised).
  private func alphaCumprod(at timestep: Int) -> Float {
    guard timestep >= 0 else {
      return 1.0  // terminal denoised state
    }
    let idx = min(timestep, alphasCumprod.count - 1)
    return alphasCumprod[max(0, idx)]
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

    case .scaledLinear(let betaStart, let betaEnd):
      // Matches HuggingFace "scaled_linear": linspace(sqrt(start), sqrt(end), T)²
      // Used by PixArt-Sigma, SD 1.x/2.x, SDXL.
      let sqrtStart = sqrt(betaStart)
      let sqrtEnd = sqrt(betaEnd)
      return (0..<trainTimesteps).map { i in
        let t = Float(i) / Float(trainTimesteps - 1)
        let sqrtBeta = sqrtStart + t * (sqrtEnd - sqrtStart)
        return sqrtBeta * sqrtBeta
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
