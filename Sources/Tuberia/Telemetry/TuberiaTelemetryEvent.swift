import Foundation
@preconcurrency import MLX

/// Cross-repo telemetry event vocabulary for SwiftTuberia's protocol seams.
///
/// Every event in this enum corresponds to a row in §5 of
/// `REQUIREMENTS-instrumentation.md`. Cases are organized in emission order
/// (lifecycle → assembly → memory → weights → LoRA → readiness → encoder →
/// scheduler → denoise loop → CFG cast → backbone → decoder → renderer →
/// anomaly side-channel → error side-channel).
///
/// The Vinetas-host adapter (`TuberiaTelemetryAdapter`) switches exhaustively
/// over this enum; adding a new case is a deliberate cross-repo contract
/// change. Adding nested enum cases (e.g. a new `AssemblyCheck`) is similarly
/// load-bearing — the host adapter will fail to compile until the new sub-case
/// is mapped to its phase suffix.
public enum TuberiaTelemetryEvent: Sendable {

  // MARK: - Lifecycle

  case pipelineConfigured(
    recipeName: String,
    encoderType: String,
    schedulerType: String,
    backboneType: String,
    decoderType: String,
    rendererType: String,
    encoderQuantization: String,
    backboneQuantization: String,
    decoderQuantization: String,
    peakMemoryBytes: UInt64,
    phasedMemoryBytes: UInt64
  )

  case pipelineStart(
    runID: UUID,
    prompt: String,
    steps: Int,
    guidanceScale: Double,
    seed: UInt32,
    width: Int,
    height: Int
  )

  case pipelineEnd(
    runID: UUID,
    totalSteps: Int,
    durationSeconds: Double,
    success: Bool
  )

  // MARK: - Assembly validation

  case assemblyCheckPassed(check: AssemblyCheck, inlet: String, outlet: String)
  case assemblyCheckFailed(check: AssemblyCheck, inlet: String, outlet: String, reason: String)

  // MARK: - Memory gate

  case memoryGateChecked(requiredBytes: UInt64, passed: Bool)

  // MARK: - Weight loading

  case weightLoadStart(role: String, componentID: String)
  case weightLoadComplete(
    role: String,
    componentID: String,
    paramCount: Int,
    totalBytes: UInt64,
    durationSeconds: Double
  )

  // MARK: - LoRA

  case loraLoadStart(
    componentID: String?,
    localPath: String?,
    scale: Double,
    activationKeyword: String?
  )
  case loraLoadComplete(adapterParamCount: Int, durationSeconds: Double)
  case loraApplied(targetLayerCount: Int)
  case loraUnapplied(restoredLayerCount: Int)

  // MARK: - Component readiness

  case componentReadinessChecked(componentID: String, ready: Bool)

  // MARK: - Text encoder handoff

  case textEncoderForwardStart(role: TextEncoderRole, promptLength: Int, maxLength: Int)
  case textEncoderForwardComplete(
    role: TextEncoderRole,
    embeddingStat: TuberiaTensorStat,
    maskStat: TuberiaTensorStat,
    durationSeconds: Double
  )

  // MARK: - Scheduler

  case schedulerConfigured(
    steps: Int,
    startTimestep: Int?,
    predictionType: String,
    timestepsHead: [Int],
    timestepsTail: [Int],
    sigmasHead: [Float],
    sigmasTail: [Float]
  )

  // MARK: - Per-step denoise

  case denoiseStepStart(
    stepIndex: Int,
    totalSteps: Int,
    timestep: Int,
    sigma: Float,
    useCFG: Bool,
    latentBeforeStat: TuberiaTensorStat
  )
  case denoiseStepComplete(
    stepIndex: Int,
    totalSteps: Int,
    timestep: Int,
    sigma: Float,
    latentAfterStat: TuberiaTensorStat,
    predictionStat: TuberiaTensorStat,
    durationSeconds: Double
  )

  // MARK: - CFG dtype cast

  case cfgDtypeCast(
    stepIndex: Int,
    fromDtype: String,
    toDtype: String,
    guidedPredictionStat: TuberiaTensorStat
  )

  // MARK: - Backbone boundary

  case backboneForwardStart(
    branch: BackboneBranch,
    conditioningStat: TuberiaTensorStat,
    latentStat: TuberiaTensorStat,
    timestep: Int
  )
  case backboneForwardComplete(
    branch: BackboneBranch,
    predictionStat: TuberiaTensorStat,
    durationSeconds: Double
  )

  // MARK: - Decoder handoff

  case decoderDecodeStart(latentStat: TuberiaTensorStat, scalingFactor: Float)
  case decoderDecodeComplete(outputStat: TuberiaTensorStat, durationSeconds: Double)

  // MARK: - Renderer handoff

  case rendererRenderStart(modality: String, inputStat: TuberiaTensorStat)
  case rendererRenderComplete(outputBytes: Int, durationSeconds: Double)

  // MARK: - Anomaly side-channel

  case numericalAnomaly(phase: String, kind: AnomalyKind, stepIndex: Int?, stat: TuberiaTensorStat)

  // MARK: - Error side-channel

  case errorThrown(phase: ErrorPhase, errorDescription: String, stepIndex: Int?)

  // MARK: - Nested enums

  /// Identifies which of the six `validateAssembly` checks fired. Names match
  /// `DiffusionPipeline.validateAssembly` member checks one-for-one.
  public enum AssemblyCheck: String, Sendable, Codable, Hashable {
    /// All components non-nil.
    case completeness
    /// `outputEmbeddingDim == expectedConditioningDim`.
    case encoderToBackboneDim
    /// `maxSequenceLength == expectedMaxSequenceLength`.
    case encoderToBackboneSeq
    /// `outputLatentChannels == expectedInputChannels`.
    case backboneToDecoder
    /// Modality compatibility (today: type-system enforced; reserved if a
    /// runtime check is ever added).
    case decoderToRenderer
    /// `BidirectionalDecoder` conformance for img2img.
    case imageToImageBidirectional
  }

  /// Whether a text-encoder forward pass produced the conditional or
  /// unconditional embedding for CFG. Single-pass (non-CFG) generations only
  /// emit `.conditional`.
  public enum TextEncoderRole: String, Sendable, Codable, Hashable {
    case conditional
    /// Empty / negative prompt for CFG.
    case unconditional
  }

  /// Which arm of the CFG branching produced a backbone forward call.
  public enum BackboneBranch: String, Sendable, Codable, Hashable {
    case noCFG
    case cfgConditional
    case cfgUnconditional
  }

  /// The numerical-anomaly side-channel's discriminator. `outOfRange` uses the
  /// `TuberiaTensorStat.defaultOutOfRangeThreshold` magnitude (default `1e6`).
  public enum AnomalyKind: String, Sendable, Codable, Hashable {
    case nan
    case inf
    /// Configurable threshold; default `>1e6` in magnitude. See
    /// `TuberiaTensorStat.defaultOutOfRangeThreshold`.
    case outOfRange
  }

  /// Which phase emitted a thrown error. The adapter maps each case to a
  /// distinct phase suffix; new cases are deliberate cross-repo contract
  /// changes.
  public enum ErrorPhase: String, Sendable, Codable, Hashable {
    case assembly
    case memoryGate
    case weightLoad
    case loraLoad
    case componentReadiness
    case missingComponent
    case textEncoderForward
    case schedulerConfigure
    case schedulerStep
    case backboneForward
    case decoderDecode
    case rendererRender
    case other
  }
}
