---
title: "SwiftTuberia — Observable Telemetry Requirements"
date: 2026-05-07
status: "DRAFT — not yet scheduled"
package_version_target: "0.7.0 (minor — adds optional public API surface)"
related:
  - "../Produciesta/MULTI_REPO_TELEMETRY.md (coordination doc)"
  - "../Produciesta/TELEMETRY_STRATEGY.md (consumer architecture)"
  - "../Produciesta/ProduciestaCore/MemoryTelemetry.swift (consumer sink)"
  - "../SwiftSecuencia/Sources/SwiftSecuencia/Telemetry/SecuenciaTelemetryReporter.swift (peer pattern)"
---

# SwiftTuberia — Observable Telemetry Requirements

## 1. Background

Produciesta is the canonical consumer of intrusive-memory's Swift libraries. While diagnosing a multi-gigabyte memory leak across `VoxAlta → mlx-audio-swift → SwiftSecuencia → SwiftCompartido`, Produciesta established a repeatable **multi-repo instrumentation pattern**:

- A central sink (`MemoryTelemetry` actor in `ProduciestaCore`) captures `(episode, phase) → Snapshot` records of process RSS, Metal allocated bytes, and dependency-specific state. Output is JSON Lines for offline analysis plus a real-time stdout report.
- Each instrumented dependency owns its **own** `<Library>TelemetryEvent` enum and `<Library>TelemetryReporter` protocol. The dependency **does not import Produciesta**.
- Produciesta provides a `<Library>TelemetryAdapter` that conforms to the dependency's reporter protocol and translates events into `MemoryTelemetry.capture(episode:phase:…)` calls.
- Hosts opt in by passing a reporter into the dependency. When the reporter is `nil` (default), the dependency emits nothing and incurs no measurable overhead.

The pattern is documented in `../Produciesta/MULTI_REPO_TELEMETRY.md` and proven in `SwiftSecuencia` (events at `ForegroundAudioExporter.exportAudio`, `ScreenplayToTimelineConverter.convertToTimeline`, AVFoundation export session lifecycle), `SwiftVoxAlta`, `mlx-audio-swift`, and `SwiftCompartido`.

SwiftTuberia is currently **not instrumented**. For diffusion-pipeline workloads (PixArt, Flux-2), the bulk of resident memory and Metal pressure originates inside Tuberia: weight loading, quantization, the denoising loop, LoRA merging, and per-segment unload. A consumer running Tuberia under any non-trivial budget (mobile, multi-pipeline orchestration, batched generation) cannot today observe where memory is allocated, how long each phase takes, or whether `unloadModels()` actually returned MLX/Metal pages to the OS.

This document specifies the telemetry surface Tuberia must add so that consumers — Produciesta, model-specific packages (`pixart-swift-mlx`, `flux-2-swift-mlx`), and host applications — can observe what the pipeline is doing without taking a hard dependency on any specific telemetry framework.

## 2. Goals

1. **Observable from the outside.** A consumer wiring up a `DiffusionPipeline` can plug in a single reporter and see every phase of `loadModels`, `generate`, and `unloadModels`, plus weight-loader I/O and LoRA merge events.
2. **Zero default overhead.** When no reporter is supplied, Tuberia emits nothing. No `print`, no `os_log`, no string allocation in the hot path.
3. **No new transitive dependencies.** Tuberia is consumed by every model package in the graph. Adding `swift-log`, `swift-metrics`, `OSLog`-only APIs, or anything else not already in `Package.swift` is **out of scope**. Telemetry is delivered through a Sendable protocol; the consumer maps it to whatever sink they like.
4. **Diagnose memory in the diffusion path.** Events must include enough state (component ID, parameter count, MLX `Memory` snapshot, Metal `currentAllocatedSize`) to answer the same class of questions Produciesta asks of `SwiftVoxAlta`: did the unload actually free anything? Is a segment retaining weights past its scope?
5. **Stable contract.** Adding new event cases is a minor-version bump. Removing or reshaping a case is a major-version bump. Consumers can switch exhaustively over the enum and trust their compile to flag breakage.

## 3. Non-Goals

- **Not a logging framework.** Tuberia does not own log levels, log routing, file rotation, or sampling. Those belong to the consumer's reporter implementation.
- **Not metrics or tracing.** No counters, no histograms, no spans. The consumer can derive any of those from the event stream if they want.
- **No CLI flag.** Tuberia is a library; it has no CLI. The `--telemetry` flag lives in `produciesta-cli` and toggles whether the host wires up a reporter.
- **No automatic file output.** The reporter decides where events go. JSON-Lines export, like Produciesta's, is implemented in the consumer.
- **No instrumentation of `mlx-swift` or `swift-tokenizers`.** Both are external. Tuberia wraps them and emits Tuberia-owned events at the boundary; it does not edit upstream code.

## 4. Public API Surface

All new types live in `Sources/Tuberia/Telemetry/`. They are exported from the `Tuberia` product. `TuberiaCatalog` does **not** add a parallel surface — catalog components emit through whatever pipeline reference the orchestrator already holds.

### 4.1 Event enum

```swift
// Sources/Tuberia/Telemetry/TuberiaTelemetryEvent.swift

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
```

Notes:

- Existence of `errorThrown` does **not** swallow the error. The throw still happens; the event is a side-channel notification.
- `quantization` is a `String` (via a new `CustomStringConvertible` conformance on `QuantizationConfig`) rather than the typed enum to avoid reshuffling the existing public enum's surface in this work item.
- The `unloadSegment` case overloads "per-segment" and "post-cache-clear" by using `role: PipelineRole?`. This is intentional — fewer cases beat a parallel `gpuCacheCleared` case that always fires once and would just clutter switches.

### 4.2 Reporter protocol

```swift
// Sources/Tuberia/Telemetry/TuberiaTelemetryReporter.swift

/// Receives telemetry events from SwiftTuberia. Implementations forward events to
/// whatever sink the host application uses (Produciesta's MemoryTelemetry actor,
/// OSLog, Sentry, swift-metrics, custom JSONL writer, etc).
///
/// SwiftTuberia does not depend on, log to, or assume the existence of any specific
/// telemetry framework. If no reporter is set, no events are emitted.
///
/// ## Concurrency
///
/// `Sendable`-bounded so the pipeline (an `actor`) can hold and call across hops.
/// `capture` is `async` so a reporter that batches or writes to disk can do so
/// without blocking pipeline progress, but it must remain non-throwing — the
/// pipeline never lets telemetry fail a generation.
public protocol TuberiaTelemetryReporter: Sendable {
  func capture(_ event: TuberiaTelemetryEvent) async
}
```

A no-op reporter is provided for testing convenience:

```swift
public struct NoopTuberiaTelemetryReporter: TuberiaTelemetryReporter {
  public init() {}
  public func capture(_ event: TuberiaTelemetryEvent) async {}
}
```

### 4.3 Wiring into `DiffusionPipeline`

`DiffusionPipeline` already exposes seam setters (`setComponentReadinessService`, `setMemoryGate`). Telemetry follows the same pattern — **no init-signature change** so existing call sites compile unchanged.

```swift
extension DiffusionPipeline {
  /// Replace the telemetry reporter. Default is `nil` (no events emitted).
  /// Call before `loadModels(progress:)` to capture load events.
  public func setTelemetry(_ reporter: (any TuberiaTelemetryReporter)?) {
    self.telemetry = reporter
  }
}
```

The actor stores `var telemetry: (any TuberiaTelemetryReporter)? = nil` and emits via a private helper:

```swift
private func emit(_ event: TuberiaTelemetryEvent) async {
  guard let telemetry else { return }
  await telemetry.capture(event)
}
```

The `guard let telemetry else { return }` is the only cost in the no-reporter path: a nil-check on a stored optional. No event constructor runs because the call site computes the event lazily — see §5.

### 4.4 Wiring into `WeightLoader`

`WeightLoader` is a `Sendable` value-type struct with `static func load`. It cannot hold instance state. Two acceptable shapes:

**Option A (preferred):** add an optional `telemetry:` parameter to the static APIs, defaulting to `nil`.

```swift
public static func load(
  componentId: String,
  keyMapping: KeyMapping,
  tensorTransform: TensorTransform? = nil,
  quantization: QuantizationConfig = .asStored,
  telemetry: (any TuberiaTelemetryReporter)? = nil   // NEW
) async throws -> ModuleParameters
```

`DiffusionPipeline.loadModels` forwards its own `self.telemetry` into each `WeightLoader.load` call. Existing call sites (model-specific packages calling `WeightLoader.load` directly) continue to work — the parameter has a default.

**Option B (rejected):** task-local. `@TaskLocal static var telemetry`. Less discoverable, harder to test, and the existing seam pattern in `DiffusionPipeline` is explicit setters — staying consistent matters more than saving a parameter.

### 4.5 Wiring into `MemoryManager`

`MemoryManager.shared` is a global actor singleton. Adding a setter mirrors `SwiftCompartido`'s pattern (Produciesta's `compartidoTelemetry` is injected via `MemoryManager.shared.setTelemetry(...)`).

```swift
extension MemoryManager {
  public func setTelemetry(_ reporter: (any TuberiaTelemetryReporter)?) {
    self.telemetry = reporter
  }
}
```

`MemoryManager` emits events at:
- `registerLoaded` → no new event (covered by `segmentApplied`)
- `unregisterLoaded` → no new event (covered by `unloadSegment`)
- `clearGPUCache` → emits `unloadSegment(role: nil, ...)` with before/after Metal bytes if telemetry is set

The duplication with `DiffusionPipeline.unloadModels` is intentional: `clearGPUCache` may be called by other code paths (LoRA reloads, host eviction), and we want the Metal-released-bytes probe regardless of who called it.

## 5. Instrumentation Points (call-site guidance)

For each call site below, the implementer must:

1. Compute event payload **only** if `telemetry != nil`. Wrap with `if let telemetry { await telemetry.capture(...) }` rather than the `emit` helper when constructing the event would do non-trivial work (e.g. snapshotting MLX memory, counting parameters). The helper is fine for events whose payload is already in scope.
2. Sample MLX/Metal counters via `MemoryManager.shared.loadedComponentsMemory` and `MTLCreateSystemDefaultDevice()?.currentAllocatedSize` — the same calls Produciesta uses. Cache the device.
3. Never let a thrown reporter (it can't throw) or slow reporter (it can `await`) abort generation. The pipeline is the source of truth; telemetry is a passive observer.

| Location | Event(s) |
|---|---|
| `DiffusionPipeline.loadModels` entry | `loadStart` |
| Inside `do { try await memoryGate(peak) }` (both branches) | `memoryGate` |
| Top of the `for (segment, ...) in weightedSegments` loop | `segmentLoadStart` |
| Inside `componentReadinessService.ensureComponentReady` progress closure | `segmentDownloadProgress` (rate-limit at the call site if `overallProgress` ticks too often — caller-side throttling, not reporter-side) |
| After `WeightLoader.load` returns, before `segment.apply(weights:)` | `segmentWeightsLoaded` |
| After `MemoryManager.shared.registerLoaded` | `segmentApplied` |
| `loadModels` exit (after the loop) | `loadComplete` |
| `DiffusionPipeline.unloadModels`, per `unload()` call | `unloadSegment(role: …, componentId: …, metalAllocatedBytesBefore/After: nil)` |
| `MemoryManager.clearGPUCache` body, around `MLX.Memory.clearCache()` | `unloadSegment(role: nil, …)` with before/after Metal bytes |
| `DiffusionPipeline.generate` entry, after `isLoaded` guards | `generateStart` |
| Before encoder's first encode | `encodeStart` |
| After conditional + (optional) unconditional encode | `encodeComplete` |
| Inside the denoising loop, **per step** | `denoiseStep` |
| Before `decoder.decode(latents)` | `decodeStart` |
| After decode | `decodeComplete` |
| Before `renderer.render(...)` | `renderStart` |
| After render | `renderComplete` |
| `generate` exit | `generateComplete` |
| Top of LoRA branch in `generate` | `loraLoadStart` |
| After `LoRALoader.apply` | `loraMerge` |
| After `LoRALoader.unapply` | `loraUnmerge` |
| Every `throw PipelineError.…` site in the public API | `errorThrown` (emit before throwing; never let the emit fail the throw) |

**Per-step event volume.** A 28-step denoise emits 28 `denoiseStep` events. Reporters are expected to handle that — but Tuberia must not allocate Strings or arrays inside the loop. `denoiseStep` payload is three primitive values; the case constructor is cheap. The loop body must not call `MemoryManager.shared.loadedComponentsMemory` or other actor hops per step — that would serialize the whole denoise on the memory actor.

## 6. Performance Budget

Telemetry-disabled (`reporter == nil`):
- ≤ 1 nil-check per call site. No event allocation. Verified by `xcodebuild test` running existing performance tests at flat baseline.

Telemetry-enabled with `NoopTuberiaTelemetryReporter`:
- ≤ 1% wall-clock regression on a 28-step PixArt generation in `Tests/TuberiaGPUTests/`. Measured by adding a "with-telemetry" variant of the existing end-to-end test and asserting the delta.
- ≤ 16 KB additional resident memory (event allocation + reporter await machinery), measured via `MemoryManager.shared.loadedComponentsMemory` deltas in test.

Telemetry-enabled with a real reporter (Produciesta's adapter):
- Out of Tuberia's scope. Produciesta's adapter is responsible for not regressing. Tuberia tests use the no-op.

## 7. Errors

`TuberiaTelemetryEvent.errorThrown` carries the `PipelineError` directly. `PipelineError` is already `Sendable`. No new error type is introduced. Reporter implementations must not retain the error past their `capture` call (it may carry MLX-related associated values that need to be dropped promptly).

## 8. Testing Requirements

All new tests live under `Tests/TuberiaTests/Telemetry/` (CPU-only suite — no GPU, no real model weights).

### 8.1 Unit tests

`TuberiaTelemetryEventTests`:
- Encodable/decodable round-trip for each case (using a hand-rolled `Codable` conformance — Tuberia events do **not** ship `Codable` automatically; that's a consumer concern. This test only verifies the structural shape via a test-local `Codable` extension to catch accidental case reordering).
- `Sendable` conformance asserted by storing across an actor hop in test.

`MockTuberiaTelemetryReporter`:
- Test fixture that records every captured event into a `[TuberiaTelemetryEvent]` array. Sendable, actor-isolated. Mirrors `SwiftSecuencia`'s `MockTelemetryReporter`.

`DiffusionPipelineTelemetryTests` (uses existing `MockPipelineRecipe`):
- `loadStart` fires exactly once before `memoryGate`.
- `memoryGate` fires with `passed == true` on happy path; with `passed == false` followed by `errorThrown(phase: "loadModels.memoryGate", …)` when the gate is stubbed to throw.
- `segmentLoadStart`/`segmentWeightsLoaded`/`segmentApplied` fire in the canonical order, once per `componentIdFor` entry.
- `loadComplete.totalEstimatedBytes` equals the sum of `segmentApplied.estimatedMemoryBytes`.
- `unloadModels` emits one `unloadSegment` per role plus one terminal `unloadSegment(role: nil, …)`.
- With `reporter == nil`, no events are observable (verified via `MockTuberiaTelemetryReporter` *not* attached and a separate spy on the existing progress callback to confirm the pipeline still ran).

`GenerateTelemetryTests`:
- Stubbed encoder/backbone/decoder/renderer; no MLX work.
- `denoiseStep` event count equals `request.steps`.
- `loraLoadStart`/`loraMerge`/`loraUnmerge` fire iff `request.loRA != nil`.
- `errorThrown(phase: "generate.encode", …)` fires when encoder throws.

`WeightLoaderTelemetryTests`:
- Synthetic safetensors fixture (existing pattern in `WeightLoaderIntegrationTests`).
- Telemetry parameter defaulting to `nil` produces zero events.
- Telemetry parameter set produces `segmentWeightsLoaded` with non-zero `safetensorsFileCount` and `parameterCount`.

`MemoryManagerTelemetryTests`:
- `clearGPUCache` with reporter set emits `unloadSegment(role: nil, ...)` with non-nil Metal bytes (Metal device is available on CI macOS runners; skip on Linux if the matrix grows).

### 8.2 Performance regression test

`TelemetryOverheadTests` (gated behind a Makefile target — these are slow):
- Runs the existing `MockPipelineRecipe` end-to-end load+generate+unload twice: once with `nil` reporter, once with `NoopTuberiaTelemetryReporter()`.
- Asserts wall-clock delta under 1% over 10 trials. Failure threshold is generous because Swift Testing parallel scheduling on macos-26 is noisy; the goal is to catch a 10% regression, not a 0.5% one.
- Excluded from the default `make test` and CI runs (MLX parallelism rule still applies). Invoked manually before release.

## 9. Documentation

- `AGENTS.md` gets a new "Telemetry" section pointing at `Sources/Tuberia/Telemetry/` and noting the opt-in contract.
- `README.md` gets a 5-line consumer snippet:

  ```swift
  let pipeline = try DiffusionPipeline(recipe: recipe)
  await pipeline.setTelemetry(MyReporter())   // optional
  try await pipeline.loadModels(progress: { _, _ in })
  ```

- One example reporter implementation lives in `Tests/TuberiaTests/Telemetry/MockTuberiaTelemetryReporter.swift` and is referenced from the README.
- A short "How Produciesta consumes this" appendix in `AGENTS.md` linking to `../Produciesta/MULTI_REPO_TELEMETRY.md` so future contributors understand the multi-repo pattern, without making Tuberia depend on Produciesta in any way.

## 10. Versioning

- This work ships as `0.7.0` (minor bump). It is purely additive at the public-API level: new types, new optional setters, new defaulted parameters.
- Existing callers — `pixart-swift-mlx`, `flux-2-swift-mlx`, plus any host consuming `DiffusionPipeline` — recompile without source changes.
- No new dependencies in `Package.swift`. No new platform requirements.

## 11. Out of Scope (and rationale)

| Item | Why deferred |
|---|---|
| Per-tensor allocation events inside `WeightLoader` | Cardinality explosion (T5-XXL alone has > 600 tensors) with no diagnostic value beyond `segmentWeightsLoaded.parameterCount`. Add only if a future leak investigation needs it. |
| MLX `Memory.snapshot()` integration | Not yet exposed in the `mlx-swift` floor (`0.31.3`). Reporters that need raw MLX numbers can call upstream APIs directly via the consumer's own code — Tuberia stays out of it. |
| `OSLog`/`signpost` mirror | A sufficiently motivated reporter implementation can do this in 10 lines. Putting it in Tuberia ties the package to Apple's logging surface area; the protocol is the seam. |
| Sampling / rate-limiting inside Tuberia | Reporter responsibility. The library emits truthfully; the consumer decides what to do with the firehose. |
| Cross-pipeline correlation IDs (tracing) | Adds an init-time parameter, complicates the protocol, and Produciesta's `(episode, phase)` keying already serves the consumer's diagnostic need. Revisit if a non-Produciesta consumer asks. |
| Telemetry for `TuberiaCatalog` concrete components beyond what flows through `DiffusionPipeline` | Catalog components do not own lifecycle — the pipeline does. Emitting from inside, e.g., `T5XXLEncoder.encode` would duplicate `encodeStart`/`encodeComplete` and risks double-counting. |

## 12. Acceptance Criteria

- [ ] `Sources/Tuberia/Telemetry/{TuberiaTelemetryEvent,TuberiaTelemetryReporter,NoopTuberiaTelemetryReporter}.swift` exist and are exported from the `Tuberia` product.
- [ ] `DiffusionPipeline.setTelemetry(_:)`, `MemoryManager.setTelemetry(_:)` exist; `WeightLoader.load(…, telemetry:)` defaulted parameter exists.
- [ ] All instrumentation points in §5 emit when a reporter is attached and emit nothing when it isn't (verified by tests in §8.1).
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO` passes including the new `Tests/TuberiaTests/Telemetry/` suite.
- [ ] No new dependencies in `Package.swift`.
- [ ] `AGENTS.md` and `README.md` updated; consumer-facing snippet present.
- [ ] Produciesta engineer can wire up a `TuberiaTelemetryAdapter` in `ProduciestaCore/Telemetry/` mirroring `SecuenciaTelemetryAdapter` and observe load/generate/unload phases for a PixArt run, **without** Tuberia importing Produciesta.

## 13. Open Questions

1. **Should `setTelemetry` accept a `Sendable` closure (`@Sendable (TuberiaTelemetryEvent) async -> Void`) instead of a protocol?** Closures are simpler for one-off consumers; protocols are easier to subclass/mock. The peer libraries (`SwiftSecuencia`, `mlx-audio-swift`) all chose the protocol form. **Recommendation: match peers, ship the protocol.**
2. **Should `errorThrown.phase` be a structured enum?** It would help consumers filter without parsing strings. But every new throw site would need an enum case, dragging the pipeline's error vocabulary into the telemetry surface. **Recommendation: keep `phase: String` for now; revisit if we see a third Produciesta-class consumer asking.**
3. **Per-step `denoiseStep` — emit always or sample?** If we sample inside Tuberia, we lock the consumer into our policy. If we always emit, a 1000-step run hits the reporter 1000 times. **Recommendation: always emit; document that reporters should debounce. Mirrors what `SwiftSecuencia` does for `timelineConversionStart`.**
4. **Do we need a `pipelineId: UUID` in every event for multi-pipeline orchestrators?** Produciesta runs one pipeline at a time. A future host (model A/B testing) might run two. **Recommendation: defer. Add via init parameter in a later minor release if a real consumer materializes.**
