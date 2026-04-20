import Tuberia

// MARK: - DPMSolverScheduler Configuration

public struct DPMSolverSchedulerConfiguration: Sendable {
  /// Beta schedule defining the noise schedule.
  public let betaSchedule: BetaSchedule
  /// What the model predicts.
  public let predictionType: PredictionType
  /// Solver order (1 = first-order Euler, 2 = second-order midpoint).
  public let solverOrder: Int
  /// Total training timesteps (for beta schedule computation).
  public let trainTimesteps: Int
  /// When true, force the final denoising step to use first-order (DDIM).
  ///
  /// Matches `lower_order_final=True` in diffusers `DPMSolverMultistepScheduler`.
  /// The last step's target sigma is effectively zero (fully denoised), so the
  /// first-order formula directly returns the x0 prediction without any noise mix-in,
  /// which is the correct terminal behaviour.
  public let lowerOrderFinal: Bool

  public init(
    betaSchedule: BetaSchedule = .linear(betaStart: 0.0001, betaEnd: 0.02),
    predictionType: PredictionType = .epsilon,
    solverOrder: Int = 2,
    trainTimesteps: Int = 1000,
    lowerOrderFinal: Bool = true
  ) {
    self.betaSchedule = betaSchedule
    self.predictionType = predictionType
    self.solverOrder = solverOrder
    self.trainTimesteps = trainTimesteps
    self.lowerOrderFinal = lowerOrderFinal
  }
}
