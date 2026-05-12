# SwiftTuberia — Instrumentation Requirements

**Status:** Draft, awaiting implementation
**Pattern source:** [Vinetas `docs/INSTRUMENTATION_PLAN.md`](https://github.com/intrusive-memory/Vinetas/blob/development/docs/INSTRUMENTATION_PLAN.md) + Produciesta `Docs/TELEMETRY_IMPL_PATTERN.md`
**Host:** Vinetas (this is one of five intrusive-memory libraries instrumented as a coherent set)
**Priority:** P1 — Tuberia owns the protocol boundaries that every other library hands tensors across; instrumenting it first establishes the handoff vocabulary the downstream libs key off of.

---

## 1. Why instrument SwiftTuberia

SwiftTuberia defines the **five protocol seams** that diffusion-based generation flows through: `TextEncoder → Scheduler → Backbone → Decoder → Renderer`. It is the single library that knows about all five contracts and orchestrates the per-step denoising loop. Every "communication error between libraries" that the user wants to track down is, by definition, a handoff at one of Tuberia's seams.

The instrumentation must surface:

- **Assembly-time shape validation outcomes.** Six checks already run in `validateAssembly` (`DiffusionPipeline.swift:129–177`). Telemetry must record their pass/fail verdicts with the full inlet/outlet/reason, even when they pass — this is the ground truth for "what protocol contract was active for this run."
- **Per-boundary handoff snapshots.** Every protocol boundary (`TextEncoder.encode`, `Backbone.forward`, `Scheduler.step`, `Decoder.decode`) emits a `TensorStat` for the value that crosses the boundary. Mismatched dtypes between libraries are caught here.
- **The CFG dtype cast at line 511.** The orchestrator casts `(uncondPrediction + scale * (condPrediction - uncondPrediction)).asType(.float32)` because backbone weights are float16 and DPM-Solver divides by ~0.006 at high sigma. This cast is a known load-bearing math operation. Telemetry must capture it as a named `dtypeBoundary` event so future "why is the image gray" investigations can confirm the cast actually happened.
- **Per-step orchestration.** Tuberia's denoise loop fires one event per scheduler step at the **outer** boundary (after CFG combination, after scheduler.step). Backbone implementations (flux, pixart) fire their own internal per-step events at their forward boundary. These nest cleanly: Tuberia's `denoiseStepComplete` brackets the backbone's `ditForward`.
- **LoRA merge/unmerge boundaries.** A LoRA-merged backbone is a different mathematical object than the base — events distinguish them so downstream anomalies don't get blamed on the base model.
- **Memory gate verdicts** and **weight load completeness** at the (boundary-set) memory snapshot points.

What it must NOT surface:
- Per-attention-block events. Those live in the backbone implementations.
- The `PipelineProgress` callback (`progress: @Sendable (PipelineProgress) -> Void`) — that's UI surface, not engineering surface. Keep as-is.

---

## 2. Coexistence with existing surfaces

| Surface | Purpose | Status under this plan |
|---|---|---|
| `PipelineProgress` callback on `DiffusionPipeline.generate(request:progress:)` | UI progress (encoding %, generating step N/total, decoding, rendering) | **Keep as-is.** Public API. Telemetry adapter on the Vinetas side can correlate. |
| `ComponentReadinessService` protocol + `setComponentReadinessService(_:)` seam | Test injection point for skipping Acervo readiness checks | **Keep as-is.** Telemetry events fire alongside readiness queries. |
| `setMemoryGate(_:)` seam | Test injection point for memory validation | **Keep as-is.** Telemetry events fire after gate execution. |
| `validateAssembly` static method (`DiffusionPipeline.swift:129`) | Six shape contract checks | **Keep as-is.** Each check now also emits an `assemblyCheckPassed` / `assemblyCheckFailed` event with the full inlet/outlet payload it already constructs for `PipelineError`. |

The existing seams are well-designed — telemetry layers on top without modifying any of them.

---

## 3. Public types to add

```
Sources/Tuberia/Telemetry/
  TuberiaTelemetryEvent.swift
  TuberiaTelemetryReporter.swift
  TuberiaTensorStat.swift
```

### 3.1 `TuberiaTensorStat.swift`

This is the shared `TensorStat` payload type. The same struct will appear in flux-2-swift-mlx, pixart-swift-mlx, and SwiftVinetas's REQUIREMENTS docs, but it lives **here** because Tuberia is the lowest in the dep graph that has access to `MLXArray`. The other libs import it via `import Tuberia`.

```swift
@preconcurrency import MLX
import Foundation

public struct TuberiaTensorStat: Sendable, Codable {
    public let shape: [Int]
    public let dtype: String     // canonical MLX dtype string, e.g. "float16", "float32", "bfloat16", "int32"
    public let min: Double
    public let max: Double
    public let mean: Double
    public let std: Double
    public let hasNaN: Bool
    public let hasInf: Bool

    public init(shape: [Int], dtype: String, min: Double, max: Double, mean: Double, std: Double, hasNaN: Bool, hasInf: Bool) {
        self.shape = shape; self.dtype = dtype
        self.min = min; self.max = max; self.mean = mean; self.std = std
        self.hasNaN = hasNaN; self.hasInf = hasInf
    }

    /// Sample a tensor's statistics. Performs eight MLX reductions; all run on the device async.
    /// **Caller MUST guard:** only call this when a telemetry reporter is attached.
    /// The function itself does not check; callers are responsible for the autoclosure dance.
    public static func sample(_ array: MLXArray) -> TuberiaTensorStat {
        // Implementation must:
        //   1. Capture shape (free) and dtype string (free).
        //   2. Compute min, max, mean, std as MLX reductions.
        //   3. Compute hasNaN via .isNaN().any() and hasInf via .isInf().any().
        //   4. Call eval() exactly once at the end on a tuple of all reductions
        //      so they run as a single graph submission, then read scalars.
        //   5. Cast everything to Double before storing (so float16/float32/bfloat16
        //      tensors produce the same Codable struct).
        // See implementation in subsequent PR.
        fatalError("Implementation pending — see PR")
    }
}
```

**Cost model:** eight reductions per `sample()` call, evaluated as one graph. With telemetry off, this function must never be called — the autoclosure guard at every emission site is what makes the cost zero.

### 3.2 `TuberiaTelemetryEvent.swift`

```swift
@preconcurrency import MLX
import Foundation

public enum TuberiaTelemetryEvent: Sendable {

    // --- Lifecycle (memory-boundary events on outer) ---
    case pipelineConfigured(
        recipeName: String,
        encoderType: String, schedulerType: String, backboneType: String, decoderType: String, rendererType: String,
        encoderQuantization: String, backboneQuantization: String, decoderQuantization: String,
        peakMemoryBytes: UInt64, phasedMemoryBytes: UInt64
    )
    case pipelineStart(runID: UUID, prompt: String, steps: Int, guidanceScale: Double, seed: UInt32, width: Int, height: Int)
    case pipelineEnd(runID: UUID, totalSteps: Int, durationSeconds: Double, success: Bool)

    // --- Assembly validation (6 checks already exist in validateAssembly) ---
    case assemblyCheckPassed(check: AssemblyCheck, inlet: String, outlet: String)
    case assemblyCheckFailed(check: AssemblyCheck, inlet: String, outlet: String, reason: String)

    // --- Memory gate ---
    case memoryGateChecked(requiredBytes: UInt64, passed: Bool)

    // --- Weight loading ---
    case weightLoadStart(role: String, componentID: String)
    case weightLoadComplete(role: String, componentID: String, paramCount: Int, totalBytes: UInt64, durationSeconds: Double)

    // --- LoRA (when present) ---
    case loraLoadStart(componentID: String?, localPath: String?, scale: Double, activationKeyword: String?)
    case loraLoadComplete(adapterParamCount: Int, durationSeconds: Double)
    case loraApplied(targetLayerCount: Int)
    case loraUnapplied(restoredLayerCount: Int)

    // --- Component readiness ---
    case componentReadinessChecked(componentID: String, ready: Bool)

    // --- Text encoder handoff ---
    case textEncoderForwardStart(role: TextEncoderRole, promptLength: Int, maxLength: Int)
    case textEncoderForwardComplete(role: TextEncoderRole, embeddingStat: TuberiaTensorStat, maskStat: TuberiaTensorStat, durationSeconds: Double)

    // --- Scheduler ---
    case schedulerConfigured(steps: Int, startTimestep: Int?, predictionType: String, timestepsHead: [Int], timestepsTail: [Int], sigmasHead: [Float], sigmasTail: [Float])
    // Head/tail = first 5 + last 5; full lists for visual inspection without bloating every record.

    // --- Per-step denoise (the high-frequency event) ---
    case denoiseStepStart(stepIndex: Int, totalSteps: Int, timestep: Int, sigma: Float, useCFG: Bool, latentBeforeStat: TuberiaTensorStat)
    case denoiseStepComplete(stepIndex: Int, totalSteps: Int, timestep: Int, sigma: Float, latentAfterStat: TuberiaTensorStat, predictionStat: TuberiaTensorStat, durationSeconds: Double)

    // --- CFG dtype cast (the documented float32 cast at DiffusionPipeline.swift:511) ---
    case cfgDtypeCast(stepIndex: Int, fromDtype: String, toDtype: String, guidedPredictionStat: TuberiaTensorStat)

    // --- Backbone boundary (orchestration-side; backbone fires its own internal events) ---
    case backboneForwardStart(branch: BackboneBranch, conditioningStat: TuberiaTensorStat, latentStat: TuberiaTensorStat, timestep: Int)
    case backboneForwardComplete(branch: BackboneBranch, predictionStat: TuberiaTensorStat, durationSeconds: Double)

    // --- Decoder handoff ---
    case decoderDecodeStart(latentStat: TuberiaTensorStat, scalingFactor: Float)
    case decoderDecodeComplete(outputStat: TuberiaTensorStat, durationSeconds: Double)

    // --- Renderer handoff ---
    case rendererRenderStart(modality: String, inputStat: TuberiaTensorStat)
    case rendererRenderComplete(outputBytes: Int, durationSeconds: Double)

    // --- Anomaly side-channel (fires when hasNaN || hasInf observed inside emission) ---
    case numericalAnomaly(phase: String, kind: AnomalyKind, stepIndex: Int?, stat: TuberiaTensorStat)

    // --- Error side-channel ---
    case errorThrown(phase: ErrorPhase, errorDescription: String, stepIndex: Int?)

    public enum AssemblyCheck: String, Sendable {
        case completeness            // all components non-nil
        case encoderToBackboneDim    // outputEmbeddingDim == expectedConditioningDim
        case encoderToBackboneSeq    // maxSequenceLength == expectedMaxSequenceLength
        case backboneToDecoder       // outputLatentChannels == expectedInputChannels
        case decoderToRenderer       // modality compat (today: type-system; reserved if runtime check added)
        case imageToImageBidirectional  // BidirectionalDecoder conformance for img2img
    }

    public enum TextEncoderRole: String, Sendable {
        case conditional
        case unconditional   // empty/negative prompt for CFG
    }

    public enum BackboneBranch: String, Sendable {
        case noCFG
        case cfgConditional
        case cfgUnconditional
    }

    public enum AnomalyKind: String, Sendable {
        case nan
        case inf
        case outOfRange       // configurable threshold; default >1e6 in magnitude
    }

    public enum ErrorPhase: String, Sendable {
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
```

### 3.3 `TuberiaTelemetryReporter.swift`

```swift
public protocol TuberiaTelemetryReporter: Sendable {
    func capture(_ event: TuberiaTelemetryEvent) async
}

public struct NoopTuberiaTelemetryReporter: TuberiaTelemetryReporter {
    public init() {}
    public func capture(_ event: TuberiaTelemetryEvent) async {}
}
```

---

## 4. Injection points

### 4.1 `DiffusionPipeline` (the actor, `Sources/Tuberia/Pipeline/DiffusionPipeline.swift:14`)

```swift
extension DiffusionPipeline {
    public func setTelemetry(_ reporter: (any TuberiaTelemetryReporter)?) {
        self.telemetry = reporter
    }
}
```

A new private ivar: `private var telemetry: (any TuberiaTelemetryReporter)? = nil`. All emission sites are inside actor-isolated methods, so no synchronization concern.

### 4.2 `LoRALoader` (struct, `Sources/Tuberia/Pipeline/LoRALoader.swift`)

Two static methods get a defaulted telemetry parameter:

```swift
public static func loadAdapterWeights(
    config: LoRAConfig,
    keyMapping: KeyMapping,
    telemetry: (any TuberiaTelemetryReporter)? = nil  // ← added
) async throws -> ModuleParameters

public static func apply(
    adapterWeights: ModuleParameters,
    to baseWeights: ModuleParameters,
    scale: Float,
    telemetry: (any TuberiaTelemetryReporter)? = nil  // ← added
) -> ModuleParameters
```

### 4.3 `WeightLoader` (struct, `Sources/Tuberia/Infrastructure/WeightLoader.swift`)

```swift
public static func load(
    componentId: String,
    keyMapping: KeyMapping,
    telemetry: (any TuberiaTelemetryReporter)? = nil  // ← added
) async throws -> ModuleParameters
```

### 4.4 `MemoryManager` (`Sources/Tuberia/Infrastructure/MemoryManager.swift`)

`MemoryManager.shared.hardValidate(requiredBytes:)` becomes:

```swift
public func hardValidate(
    requiredBytes: UInt64,
    telemetry: (any TuberiaTelemetryReporter)? = nil  // ← added
) async throws
```

The `DiffusionPipeline.memoryGate` closure default is updated to pass the actor's `telemetry` through.

---

## 5. Per-event emission spec

| Event | Emission site (file:line) | Hot-path notes |
|---|---|---|
| `pipelineConfigured` | `DiffusionPipeline.init(recipe:)` end (`DiffusionPipeline.swift:~125`) | Once per pipeline. Free. |
| `pipelineStart` | `generate(...)` entry (`DiffusionPipeline.swift:325`) | Once. |
| `pipelineEnd` | `generate(...)` exit (success path) and error path via `defer` | Once. Carries success Bool. |
| `assemblyCheckPassed` | After each check in `validateAssembly` (`DiffusionPipeline.swift:134–172`) | 6 emissions per pipeline init. |
| `assemblyCheckFailed` | Inside each `throw PipelineError.incompatibleComponents(...)` block (before throw) | 0–6 per init. |
| `memoryGateChecked` | After `try await memoryGate(peak)` (`DiffusionPipeline.swift:~226`) | Once per `loadModels`. |
| `weightLoadStart` / `weightLoadComplete` | Around each `WeightLoader.load` invocation | One pair per component (typically 3: encoder, backbone, decoder). |
| `loraLoadStart` / `loraLoadComplete` | Around `LoRALoader.loadAdapterWeights` (`DiffusionPipeline.swift:346`) | 0 or 1 pair per generate. |
| `loraApplied` | After `try backbone.apply(weights: mergedWeights)` (`:359`) | 0 or 1. |
| `loraUnapplied` | After the deferred unmerge (search post-loop region) | 0 or 1. |
| `componentReadinessChecked` | Inside `loadModels` per component | n per `loadModels`. |
| `textEncoderForwardStart` / `Complete` | Around `encoder.encode(encoderInput)` (`:380`) and `encoder.encode(uncondInput)` (`:398`) | 1 or 2 pairs per generate. `embeddingStat` and `maskStat` sampled inside `@autoclosure` guard. |
| `schedulerConfigured` | After `scheduler.configure(...)` returns (`:475`) | Once per generate. `timestepsHead/Tail` = first 5 + last 5 of `plan.timesteps`; same for sigmas. |
| `denoiseStepStart` | Beginning of each loop iteration (`:480`) | n events (n = step count, typically 4–50). Carries `latentBeforeStat`. |
| `denoiseStepComplete` | After `eval(latents)` (`:545`) | n events. Carries `latentAfterStat` and `predictionStat`. |
| `cfgDtypeCast` | Immediately after the `.asType(.float32)` cast (`:511` for CFG branch, `:529` for non-CFG) | n events. Stat sampled on `guidedPrediction` post-cast. |
| `backboneForwardStart` / `Complete` | Around each `backbone.forward(...)` call (`:502, :503, :529`) | 1 or 2 pairs per step. **Cost-critical** — see hot-path discipline below. |
| `decoderDecodeStart` / `Complete` | Around `decoder.decode(latents)` (`:~552`) | One pair per generate. |
| `rendererRenderStart` / `Complete` | Around `renderer.render(...)` (`:~560`) | One pair per generate. |
| `numericalAnomaly` | Inside `TuberiaTensorStat.sample` post-construction, if `hasNaN || hasInf || max.magnitude > anomalyThreshold` | Fires from inside any emission that sampled a tensor. Out-of-band side-channel. |
| `errorThrown` | Every `throw` in this library — lines 134, 144, 154, 169 (assembly), 227, 230 (loadModels), 327, 330, 333 (missing component), 382, 400 (encoding), 439, 463, 538 (generation), 554 (decoding), 563 (rendering), plus `LoRALoader.swift:44`, `WeightLoader.swift:51, 60, 98, 100, 102` | Each `errorThrown` emit immediately before the throw. |

### Hot-path discipline

The denoise loop is the inner loop of the entire library. Five events fire per step in the CFG path: `denoiseStepStart`, `backboneForwardStart×2`, `backboneForwardComplete×2`, `cfgDtypeCast`, `denoiseStepComplete`. With CFG off, this drops to 3 per step.

Each event that carries a `TuberiaTensorStat` performs 8 MLX reductions. Per step in CFG: ~6 stats × 8 reductions = 48 reductions/step. **All must be guarded by `@autoclosure` so they never execute when telemetry is nil.**

Emission template for in-loop sites:

```swift
if let telemetry {
    let stat = TuberiaTensorStat.sample(latents)
    await telemetry.capture(.denoiseStepComplete(
        stepIndex: stepIndex, totalSteps: totalSteps,
        timestep: timestep, sigma: plan.sigmas[stepIndex],
        latentAfterStat: stat,
        predictionStat: TuberiaTensorStat.sample(prediction),
        durationSeconds: stepDuration
    ))
}
```

Do **not** use `@autoclosure` parameter on the reporter protocol's `capture` method — the protocol stays clean. Guard at the call site.

---

## 6. Adapter mapping (Vinetas host side)

`TuberiaTelemetryAdapter` at `Vinetas/Telemetry/Adapters/TuberiaTelemetryAdapter.swift`:

| Event | Sink phase | Memory snapshot? | Payload notes |
|---|---|---|---|
| `pipelineConfigured` | `tuberia_pipeline_configured` | no | Carry recipe name + quantization triple |
| `pipelineStart` | `tuberia_pipeline_start` | **yes** | Per §3.1 memory boundary set |
| `pipelineEnd` | `tuberia_pipeline_end` | **yes** | Per §3.1 |
| `assemblyCheckPassed` | `tuberia_assembly_pass_<check>` | no | One sub-phase per `AssemblyCheck` case |
| `assemblyCheckFailed` | `tuberia_assembly_FAIL_<check>` | no | Includes `reason` |
| `memoryGateChecked` | `tuberia_memory_gate_<passed/failed>` | no | Carries `requiredBytes` |
| `weightLoadStart` | `tuberia_weight_load_start` | no | role + componentID |
| `weightLoadComplete` | `tuberia_weight_load_complete` | **yes** | Per INSTRUMENTATION_PLAN §3.1 |
| `loraLoadStart` | `tuberia_lora_load_start` | no | |
| `loraLoadComplete` | `tuberia_lora_load_complete` | no | |
| `loraApplied` / `loraUnapplied` | `tuberia_lora_apply` / `tuberia_lora_unapply` | no | |
| `componentReadinessChecked` | `tuberia_readiness_<ready/missing>` | no | |
| `textEncoderForwardStart` | `tuberia_encode_start_<role>` | no | |
| `textEncoderForwardComplete` | `tuberia_encode_complete_<role>` | no | TensorStat → Payload |
| `schedulerConfigured` | `tuberia_scheduler_configured` | no | head/tail in payload |
| `denoiseStepStart` | `tuberia_denoise_step_start` | no | stepIndex in Snapshot.stepIndex |
| `denoiseStepComplete` | `tuberia_denoise_step_complete` | no | TensorStats in Payload |
| `cfgDtypeCast` | `tuberia_cfg_cast_<from>_to_<to>` | no | e.g. `tuberia_cfg_cast_float16_to_float32` |
| `backboneForwardStart` / `Complete` | `tuberia_backbone_forward_start_<branch>` / `..._complete_<branch>` | no | One sub-phase per `BackboneBranch` |
| `decoderDecodeStart` / `Complete` | `tuberia_decode_start` / `tuberia_decode_complete` | no | |
| `rendererRenderStart` / `Complete` | `tuberia_render_start` / `tuberia_render_complete` | no | |
| `numericalAnomaly` | `tuberia_anomaly_<kind>` | no | **Critical signal — adapter MAY also push a duplicate event with elevated log level** |
| `errorThrown` | `tuberia_error_<phase>` | no | |

Adapter **switches exhaustively**. New `AssemblyCheck` or `ErrorPhase` case becomes a compile error on the host.

---

## 7. Tests

Add to `Tests/SwiftTuberiaTests/`:

| Test | Purpose |
|---|---|
| `TuberiaTelemetryAssemblyTests` | Build a recipe with deliberately mismatched components (encoder.outputEmbeddingDim ≠ backbone.expectedConditioningDim). Assert `assemblyCheckFailed(.encoderToBackboneDim, ...)` fires **before** the `PipelineError` throws. Build a valid recipe; assert all six `assemblyCheckPassed` events fire. |
| `TuberiaTelemetryDenoiseLoopTests` | Use a `MockBackbone`/`MockScheduler` that return deterministic tensors. Run 4 steps. Assert: 4× `denoiseStepStart`, 4× `denoiseStepComplete`, stepIndex monotone, `latentAfterStat.shape` matches the configured latent shape. |
| `TuberiaTelemetryCFGCastTests` | Drive a CFG run; assert `cfgDtypeCast(fromDtype: "float16", toDtype: "float32", ...)` fires per step with `guidedPredictionStat.dtype == "float32"`. |
| `TuberiaTelemetryAnomalyTests` | Inject a `MockBackbone` that returns a tensor containing NaN at step 2. Assert: (1) `denoiseStepComplete` for step 2 carries `predictionStat.hasNaN == true`; (2) `numericalAnomaly(phase: "backbone_forward_complete", kind: .nan, stepIndex: 2, ...)` is emitted within the same step boundary. |
| `TuberiaTelemetryNoopOverheadTests` | Generate 4 steps with `nil` reporter and 4 steps with `NoopTuberiaTelemetryReporter`. Both wall-clock medians within ±2% over 30 iterations. **Critical**: this is what proves the `@autoclosure` guard works. |
| `TuberiaTelemetryLoRATests` | Run a generation with a LoRA config; assert `loraLoadStart/Complete`, `loraApplied`, and `loraUnapplied` all fire in correct order. |

The first three are the load-bearing tests for the "communication errors between libraries" goal.

---

## 8. Out of scope

- Per-attention-block events. They live in the backbone implementation libs.
- Renderer-internal sampling (image encoding, audio encoding). Rendered output byte count is enough.
- `cgImageToMLXArray` helper (`DiffusionPipeline.swift:446`) — this is an internal utility, the input/output stats are captured by adjacent decoder events.
- Telemetry around `eval(latents)` (`:545`). Eval is an MLX-internal trigger; its cost shows up in the next event's `durationSeconds` field.

---

## 9. Versioning

**Minor** version bump (additive: new types, new optional parameters, new setter, new public sample method). Pin floor: `0.7.0` post-release.

---

## 10. Implementation checklist

- [ ] Add `Sources/Tuberia/Telemetry/TuberiaTensorStat.swift` per §3.1 (with real MLX-reduction implementation, not the fatalError stub)
- [ ] Add `Sources/Tuberia/Telemetry/TuberiaTelemetryEvent.swift` per §3.2
- [ ] Add `Sources/Tuberia/Telemetry/TuberiaTelemetryReporter.swift` per §3.3
- [ ] Add `setTelemetry(_:)` to `DiffusionPipeline`
- [ ] Add defaulted `telemetry:` parameter to `LoRALoader.loadAdapterWeights`, `LoRALoader.apply`, `WeightLoader.load`, `MemoryManager.hardValidate`
- [ ] Wire all emission sites per §5; each `throw` paired with a preceding `errorThrown` emit
- [ ] Add `numericalAnomaly` side-channel inside `TuberiaTensorStat.sample` post-construction
- [ ] Add tests per §7
- [ ] Run baseline overhead test; record in PR description
- [ ] Tag release with `MINOR` bump (this PR establishes the `TuberiaTensorStat` type that flux-2-swift-mlx and pixart-swift-mlx will import — ship before those libs add their instrumentation)
