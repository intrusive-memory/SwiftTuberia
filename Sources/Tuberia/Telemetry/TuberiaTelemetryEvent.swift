import Foundation

/// Telemetry events emitted by SwiftTuberia during pipeline lifecycle and generation.
///
/// All cases are `Sendable`. Consumers are expected to switch exhaustively;
/// adding a case is a minor-version bump, removing one is major.
public enum TuberiaTelemetryEvent: Sendable {

  // MARK: Pipeline lifecycle

  /// Fired once at the start of `DiffusionPipeline.loadModels`, before the memory gate runs.
  case loadStart(
    peakMemoryBytes: UInt64,
    phasedMemoryBytes: UInt64,
    availableMemoryBytes: UInt64
  )

  /// Fired after the memory gate decides. `passed == false` immediately precedes a thrown
  /// `PipelineError.insufficientMemory`; consumers can use this to record near-misses too.
  case memoryGate(
    requiredBytes: UInt64,
    availableBytes: UInt64,
    passed: Bool
  )

  /// Fired once per weighted segment, before `componentReadinessService.ensureComponentReady`.
  case segmentLoadStart(
    role: PipelineRole,
    componentId: String,
    quantization: String,    // QuantizationConfig.description; string-typed to keep the enum cheap
    estimatedBytes: UInt64
  )

  /// Fired periodically during Acervo download (only when a download actually occurs).
  /// `overallProgress` is the same `[0, 1]` fraction that flows into the existing
  /// `progress` callback. Reporter implementations should rate-limit if needed.
  case segmentDownloadProgress(
    role: PipelineRole,
    componentId: String,
    overallProgress: Double
  )

  /// Fired after `WeightLoader.load` returns, before `segment.apply(weights:)`.
  case segmentWeightsLoaded(
    role: PipelineRole,
    componentId: String,
    safetensorsFileCount: Int,
    parameterCount: Int,
    bytesAfter: UInt64       // MemoryManager.shared.loadedComponentsMemory snapshot
  )

  /// Fired after `segment.apply(weights:)` succeeds and the component is registered
  /// with `MemoryManager`. Marks the segment as live.
  case segmentApplied(
    role: PipelineRole,
    componentId: String,
    estimatedMemoryBytes: UInt64
  )

  /// Fired once `DiffusionPipeline.loadModels` returns successfully.
  case loadComplete(
    durationSeconds: Double,
    totalEstimatedBytes: UInt64
  )

  /// Fired once per segment from `DiffusionPipeline.unloadModels`, then once with
  /// `role == nil` after `MemoryManager.shared.clearGPUCache()`. The `clearGPUCache`
  /// case is the canonical "did Metal release pages?" probe.
  case unloadSegment(
    role: PipelineRole?,                 // nil ⇒ post-clearGPUCache marker
    componentId: String?,
    metalAllocatedBytesBefore: UInt64?,
    metalAllocatedBytesAfter: UInt64?
  )

  // MARK: Generation

  /// Fired at the start of `generate(request:progress:)`, after `isLoaded` guards pass.
  case generateStart(
    promptCharacterCount: Int,
    width: Int,
    height: Int,
    steps: Int,
    guidanceScale: Float,
    hasReferenceImages: Bool,
    hasLoRA: Bool,
    seed: UInt32             // resolved seed, after random fallback
  )

  /// Fired before the encoder runs.
  case encodeStart(useCFG: Bool, unconditionalStrategy: String)

  /// Fired after both conditional and (if any) unconditional encodings complete.
  case encodeComplete(
    durationSeconds: Double,
    conditionalEmbeddingShape: [Int],
    hasUnconditional: Bool
  )

  /// Fired once per denoising step — keep payload tiny, this fires `steps` times.
  /// Reporters with high-cardinality concerns should debounce.
  case denoiseStep(
    stepIndex: Int,           // 0-based
    totalSteps: Int,
    timestep: Int
  )

  /// Fired before `decoder.decode(latents)`.
  case decodeStart(latentShape: [Int])

  /// Fired after decoding, before rendering.
  case decodeComplete(durationSeconds: Double)

  /// Fired before `renderer.render(decodedOutput)`.
  case renderStart

  /// Fired after rendering succeeds.
  case renderComplete(durationSeconds: Double)

  /// Fired once `generate` returns. `totalDuration` matches the value embedded in
  /// `DiffusionGenerationResult.duration`.
  case generateComplete(
    totalDurationSeconds: Double,
    seed: UInt32,
    steps: Int
  )

  // MARK: LoRA

  /// Fired before `LoRALoader.loadAdapterWeights`.
  case loraLoadStart(
    adapterPath: String,
    scale: Float,
    activationKeyword: String?
  )

  /// Fired after adapter weights are loaded but before they're merged into the backbone.
  case loraMerge(
    adapterParameterCount: Int,
    baseParameterCount: Int,
    scale: Float
  )

  /// Fired after `LoRALoader.unapply` restores base weights at the end of `generate`.
  case loraUnmerge(scale: Float)

  // MARK: Errors

  /// Fired immediately before any `PipelineError` is thrown out of Tuberia public API.
  /// Lets reporters classify failure modes without instrumenting every call site.
  case errorThrown(
    phase: String,            // e.g. "loadModels.weightLoad", "generate.denoise.step.5"
    error: PipelineError
  )
}
