---
source: REQUIREMENTS-telemetry.md
generated_by: mission-supervisor breakdown
refined_by: mission-supervisor refine (passes 1–4)
package_version_target: our next minor release version (additive public API)
status: ready-to-execute
feature_name: OPERATION STETHOSCOPE WATCHTOWER
starting_point_commit: 37aeb415c9236983ad09296f8adced44be9df4a2
mission_branch: mission/stethoscope-watchtower/01
iteration: 1
---

# EXECUTION_PLAN.md — SwiftTuberia Observable Telemetry

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

## Mission Summary

Add an opt-in telemetry surface to SwiftTuberia so consumers (Produciesta, model-specific MLX packages, host apps) can observe `loadModels`, `generate`, `unloadModels`, weight-loader I/O, LoRA merge events, and Metal-allocated-bytes deltas — without Tuberia taking a dependency on any specific telemetry framework. Pattern mirrors the proven multi-repo seam used in SwiftSecuencia, mlx-audio-swift, SwiftCompartido. Default path emits nothing; perf budget is ≤1% wall-clock regression with a no-op reporter.

Source: `REQUIREMENTS-telemetry.md`

## Decisions Locked from Source §13

The four open questions in the requirements doc are resolved as follows; sorties below assume these decisions and must not relitigate them:

1. **Reporter shape**: `protocol TuberiaTelemetryReporter` (matches SwiftSecuencia / mlx-audio-swift peers). Closure form rejected.
2. **`errorThrown.phase`**: `String`. Not a structured enum. Revisit only if a third Produciesta-class consumer surfaces.
3. **`denoiseStep`**: emit on every step. No internal sampling. Reporters debounce.
4. **`pipelineId: UUID`**: deferred. Not part of this minor-version bump.

## Project Conventions Locked

- **Test framework**: `import Testing` (Swift Testing). All existing `TuberiaTests` files use it; new files match.
- **Test target placement**: All new tests go in `TuberiaTests/Telemetry/` (CPU-only). `TuberiaGPUTests` is not extended.
- **MockPipelineRecipe target visibility**: `MockPipelineRecipe` lives in `TuberiaGPUTests` and is **not visible** from `TuberiaTests`. New telemetry tests follow the existing in-target pattern (see `RecipeRoleMapTests.swift` line 134 `private struct ReversibleRecipe` and `MemoryGuardTests.swift` line 134 `private struct MemGuardRecipe`): each test file declares a `private` CPU-only stub recipe conforming to `PipelineRecipe`.
- **xcodebuild destination**: `'platform=macOS,arch=arm64'` everywhere (Apple Silicon only).
- **Test parallelism**: every `xcodebuild test` invocation passes `-parallel-testing-enabled NO` (MLX Metal-stream race rule).
- **Build commands**: `xcodebuild` only; never `swift build` / `swift test`.

## Work Units

| Work Unit   | Directory | Sorties | Layer | Dependencies |
|-------------|-----------|---------|-------|--------------|
| SwiftTuberia | .         | 13      | 0     | none         |

The mission is a single Swift Package; everything lives in this repo.

## Priority & Parallelism Structure

**Critical path** (longest dependency chain): Sortie 1 → Sortie 3 → Sortie 5 → Sortie 6 → Sortie 11 → Sortie 13 (length: 6 sorties).

**Parallel execution groups** (subject to the build-only-on-supervisor rule below):

- **Group A** — fires immediately after Sortie 1 completes:
  - Sortie 2 (test infra) — supervising agent only (build/test verification)
  - Sortie 3 (DiffusionPipeline load/unload instrumentation) — supervising agent only (build verification)
  - Sortie 9 (WeightLoader + MemoryManager instrumentation) — supervising agent only (build verification)
- **Group B** — fires after Sortie 2 + 3 complete:
  - Sortie 4 (DiffusionPipeline lifecycle tests) — supervising agent only (xcodebuild test)
  - Sortie 5 (generate happy path + denoiseStep instrumentation) — supervising agent only (build verification)
  - Sortie 12 (documentation) — **sub-agent candidate** (no build, no Swift code touched). Eligible for Agent dispatch when its deps (1, 3) are done.
- **Group C** — fires after Sortie 5 completes:
  - Sortie 6 (LoRA + errorThrown instrumentation) — supervising agent only
  - Sortie 7 (generate / denoiseStep tests) — supervising agent only
- **Group D** — fires after Sortie 6 + 9 + 2 complete:
  - Sortie 8 (LoRA / errorThrown tests) — supervising agent only
  - Sortie 10 (WeightLoader + MemoryManager tests) — supervising agent only
- **Group E** — fires once 3, 5, 6, 9 are all complete:
  - Sortie 11 (performance regression suite + Makefile) — supervising agent only
- **Group F** — final:
  - Sortie 13 (acceptance verification) — supervising agent only

**Agent allocation**: 1 supervising agent + up to 1 sub-agent (only Sortie 12 qualifies). Realistic concurrency in this mission is **dominated by build/test verification** — virtually every sortie ends in `xcodebuild`, which the rule restricts to the supervising agent. This is honestly reported, not artificially inflated.

**Missed opportunities** (would require splitting authoring vs. verification into separate sorties):
- Test-writing sorties (4, 7, 8, 10) could in principle have a sub-agent author the test file while the supervising agent runs `xcodebuild test`. We chose **not** to split these — the verification step is small and bundling it preserves the "one objective per sortie" sergeant principle.

---

## Sortie 1: Telemetry foundation types

**Priority**: 36.5 — highest. Foundation; transitively blocks all 12 remaining sorties.

**Entry criteria**:
- [ ] First sortie — no prerequisites.
- [ ] Working tree is clean and on the mission branch.

**Tasks**:
1. Create `Sources/Tuberia/Telemetry/TuberiaTelemetryEvent.swift` with the full `public enum TuberiaTelemetryEvent: Sendable` exactly as specified in REQUIREMENTS-telemetry.md §4.1 (lifecycle, generation, LoRA, errors).
2. Create `Sources/Tuberia/Telemetry/TuberiaTelemetryReporter.swift` with `public protocol TuberiaTelemetryReporter: Sendable { func capture(_ event: TuberiaTelemetryEvent) async }` per §4.2.
3. Create `Sources/Tuberia/Telemetry/NoopTuberiaTelemetryReporter.swift` exposing `public struct NoopTuberiaTelemetryReporter: TuberiaTelemetryReporter` with `public init()` and an empty `capture` body.
4. In the existing `Sources/Tuberia/Types/QuantizationConfig.swift`, add `extension QuantizationConfig: CustomStringConvertible { public var description: String { … } }` so the `quantization: String` payload of `segmentLoadStart` can be populated without reshaping the enum.
5. Confirm the new types are visible from the `Tuberia` product — declarations are `public`, no accidental `internal`, no nested-in-private-type placement.

**Exit criteria**:
- [ ] `xcodebuild build -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO` succeeds with zero new warnings on the new files.
- [ ] `grep -r "import Produciesta\|import ProduciestaCore" Sources/Tuberia/` returns no matches.
- [ ] `test -f Sources/Tuberia/Telemetry/TuberiaTelemetryEvent.swift && test -f Sources/Tuberia/Telemetry/TuberiaTelemetryReporter.swift && test -f Sources/Tuberia/Telemetry/NoopTuberiaTelemetryReporter.swift` returns 0.
- [ ] `grep -nE "^public " Sources/Tuberia/Telemetry/TuberiaTelemetryEvent.swift Sources/Tuberia/Telemetry/TuberiaTelemetryReporter.swift Sources/Tuberia/Telemetry/NoopTuberiaTelemetryReporter.swift` shows `public enum`, `public protocol`, and `public struct` respectively.
- [ ] `grep -n "extension QuantizationConfig: CustomStringConvertible" Sources/Tuberia/Types/QuantizationConfig.swift` returns 1 line.

---

## Sortie 2: Test infrastructure (mock reporter + event tests)

**Priority**: 21.5 — second-highest. Foundation for every test sortie that follows.

**Entry criteria**:
- [ ] Sortie 1 exit criteria all checked.

**Tasks**:
1. Create `Tests/TuberiaTests/Telemetry/MockTuberiaTelemetryReporter.swift` — an `actor`-isolated, `Sendable` recorder with a public `events` accessor returning `[TuberiaTelemetryEvent]` (mirror SwiftSecuencia's `MockTelemetryReporter`). Conforms to `TuberiaTelemetryReporter`. File-private to the test target (no `@testable import` needed by this file itself).
2. Create `Tests/TuberiaTests/Telemetry/TuberiaTelemetryEventTests.swift` using **Swift Testing** (`import Testing`, `@Test`/`#expect`/`#require`) covering:
   - (a) `Sendable` conformance: store an event in a `let` and hand it across an `await actor.method(_:)` boundary.
   - (b) Codable round-trip for every case via a *test-local* `Codable` extension on `TuberiaTelemetryEvent` declared **inside the test file** (production code does NOT ship `Codable`).
3. Both files must use `import Testing` (not XCTest) — every existing `Tests/TuberiaTests/*.swift` uses Swift Testing; do not introduce a new framework.

**Exit criteria**:
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO -only-testing:TuberiaTests/TuberiaTelemetryEventTests` passes.
- [ ] `grep -l "import Testing" Tests/TuberiaTests/Telemetry/*.swift` lists both new files.
- [ ] `grep -rn "extension TuberiaTelemetryEvent: Codable\|: Codable" Sources/Tuberia/Telemetry/` returns no matches (Codable does not leak into Sources).
- [ ] `MockTuberiaTelemetryReporter` is declared `public` or `internal` (not `private`/`fileprivate`) so subsequent test files in the same target can use it; verify with `grep -n "actor MockTuberiaTelemetryReporter\|class MockTuberiaTelemetryReporter" Tests/TuberiaTests/Telemetry/MockTuberiaTelemetryReporter.swift`.

---

## Sortie 3: DiffusionPipeline load/unload instrumentation

**Priority**: 29 — high. Establishes the actor-side seam pattern (stored var + `setTelemetry` + `emit` helper) reused by Sorties 5, 6, 9.

**Entry criteria**:
- [ ] Sortie 1 exit criteria checked (event types and reporter protocol exist).

**Tasks**:
1. In `Sources/Tuberia/Pipeline/DiffusionPipeline.swift`, add `var telemetry: (any TuberiaTelemetryReporter)? = nil` (actor-isolated stored property, sibling to the existing `componentReadinessService` and `memoryGate` seams around lines 54–67).
2. Add `public func setTelemetry(_ reporter: (any TuberiaTelemetryReporter)?) { self.telemetry = reporter }` next to the existing `setComponentReadinessService` / `setMemoryGate` setters (around lines 183–191). No init-signature change.
3. Add private async helper:
   ```swift
   private func emit(_ event: TuberiaTelemetryEvent) async {
     guard let telemetry else { return }
     await telemetry.capture(event)
   }
   ```
4. Wire the **load-side** emit sites per REQUIREMENTS §5:
   - `loadStart` — at `loadModels(progress:)` entry, before the memory gate.
   - `memoryGate(passed: true)` — after the gate's `do { try await memoryGate(peak) }` succeeds.
   - `memoryGate(passed: false)` then `errorThrown(phase: "loadModels.memoryGate", error: …)` — in the `catch` branch, **before** rethrowing.
   - `segmentLoadStart` — at the top of the `for (segment, …) in weightedSegments` loop.
   - `segmentDownloadProgress` — inside `componentReadinessService.ensureComponentReady`'s progress closure, **rate-limited at the call site** (e.g. only emit if `overallProgress` has advanced ≥ 0.01 since the last emit, tracked in a local var).
   - `segmentWeightsLoaded` — after `WeightLoader.load` returns, before `segment.apply(weights:)`.
   - `segmentApplied` — after `MemoryManager.shared.registerLoaded`.
   - `loadComplete` — at `loadModels` exit (after the loop), with `durationSeconds` measured from the entry, `totalEstimatedBytes` summed from per-segment `estimatedMemoryBytes`.
5. Wire the **unload-side** emit sites:
   - `unloadSegment(role: <role>, componentId: <id>, metalAllocatedBytesBefore: nil, metalAllocatedBytesAfter: nil)` — once per `unload()` call inside `unloadModels` (around line 299–308). Metal byte fields stay `nil` here; the post-cache-clear `unloadSegment(role: nil, …)` with Metal byte deltas is wired in Sortie 9 inside `MemoryManager.clearGPUCache`.
6. For events whose payload requires non-trivial work (parameter counting, MLX memory snapshots), guard payload construction inside `if let telemetry { … }` rather than calling `emit` unconditionally. The `emit` helper is fine for events whose payload is already in scope.
7. Do **not** wire any generate-side, LoRA-side, or `errorThrown` emits (except the single `loadModels.memoryGate` errorThrown above). Those belong to Sorties 5 and 6.
8. Do **not** forward `self.telemetry` into `WeightLoader.load(…)` calls yet — the `telemetry:` parameter is added in Sortie 9, so wiring forwarding here would cause a compile error.

**Exit criteria**:
- [ ] `xcodebuild build -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO` succeeds.
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO -only-testing:TuberiaTests` passes (existing pre-telemetry tests, no regressions).
- [ ] `grep -nE "await emit\(\.|await telemetry\?.capture\(|if let telemetry" Sources/Tuberia/Pipeline/DiffusionPipeline.swift | wc -l` returns ≥ 9 (loadStart + memoryGate ×2 branches + segmentLoadStart + segmentDownloadProgress + segmentWeightsLoaded + segmentApplied + loadComplete + per-segment unloadSegment).
- [ ] `grep -n "setTelemetry" Sources/Tuberia/Pipeline/DiffusionPipeline.swift` returns the new public setter.
- [ ] `grep -nE "private func emit\(_ event: TuberiaTelemetryEvent\)" Sources/Tuberia/Pipeline/DiffusionPipeline.swift` returns 1 line.

---

## Sortie 4: DiffusionPipeline lifecycle tests

**Priority**: 4.5 — verification layer for Sortie 3. Mid-run safety net.

**Entry criteria**:
- [ ] Sortie 2 exit criteria checked (`MockTuberiaTelemetryReporter` exists and is reachable from this target).
- [ ] Sortie 3 exit criteria checked.

**Tasks**:
1. Create `Tests/TuberiaTests/Telemetry/DiffusionPipelineTelemetryTests.swift` using Swift Testing.
2. Declare a `private struct DiffusionTelemetryStubRecipe: PipelineRecipe, Sendable` inside the test file, mirroring the inline-stub pattern in `Tests/TuberiaTests/RecipeRoleMapTests.swift` (`ReversibleRecipe`) and `Tests/TuberiaTests/MemoryGuardTests.swift` (`MemGuardRecipe`). The stub uses CPU-only no-op encoder/scheduler/backbone/decoder/renderer — **do not import or use** `MockPipelineRecipe` from `TuberiaGPUTests` (it is in a different test target and not visible).
3. Assert event ordering: `loadStart` precedes `memoryGate` precedes the per-segment trio (`segmentLoadStart` → `segmentWeightsLoaded` → `segmentApplied`) precedes `loadComplete`.
4. Happy path: `memoryGate.passed == true`.
5. Failure path: inject a stubbed `memoryGate` that throws via `pipeline.setMemoryGate(_:)`; assert `memoryGate(passed: false)` is followed by `errorThrown(phase: "loadModels.memoryGate", …)` and that the error still propagates to the caller.
6. Per-component asserts: `segmentLoadStart` / `segmentWeightsLoaded` / `segmentApplied` fire once per `componentIdFor` entry, in canonical order.
7. `loadComplete.totalEstimatedBytes == sum(segmentApplied.estimatedMemoryBytes)`.
8. `unloadModels` emits one `unloadSegment(role: <each>, …)` per loaded segment. (Note: the post-cache-clear `unloadSegment(role: nil, …)` is asserted in Sortie 10, not here, because `MemoryManager.clearGPUCache` instrumentation is added in Sortie 9.)
9. Default-`nil` reporter path: with no reporter set, no events are observable. Verify by attaching a `MockTuberiaTelemetryReporter` *only after* a control run, OR by spying on the existing `progress` callback to confirm the pipeline still ran without telemetry.

**Exit criteria**:
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO -only-testing:TuberiaTests/DiffusionPipelineTelemetryTests` passes.
- [ ] `grep -n "MockPipelineRecipe" Tests/TuberiaTests/Telemetry/DiffusionPipelineTelemetryTests.swift` returns no matches (in-target stub pattern only).
- [ ] `grep -nE "@Test|#expect|#require" Tests/TuberiaTests/Telemetry/DiffusionPipelineTelemetryTests.swift | wc -l` returns ≥ 8 (one per assertion above).

---

## Sortie 5: DiffusionPipeline generate happy path + denoiseStep instrumentation

**Priority**: 15 — high. On the critical path; unblocks 6, 7, 11.

**Entry criteria**:
- [ ] Sortie 3 exit criteria checked.

**Tasks**:
1. In `Sources/Tuberia/Pipeline/DiffusionPipeline.swift`, wire the generate-flow emit sites per REQUIREMENTS §5:
   - `generateStart` — after the `isLoaded` guard and the missing-component guards (around lines 327–333), with the resolved seed (post-random-fallback).
   - `encodeStart` — before the encoder's first encode call.
   - `encodeComplete` — after both conditional and (if any) unconditional encodings finish.
   - `decodeStart` — immediately before `decoder.decode(latents)`.
   - `decodeComplete` — after decoding succeeds.
   - `renderStart` — before `renderer.render(decodedOutput)`.
   - `renderComplete` — after rendering succeeds.
   - `generateComplete` — at `generate(…)` exit, with `totalDurationSeconds` matching the value embedded in the returned `DiffusionGenerationResult.duration`.
2. Wire `denoiseStep(stepIndex:totalSteps:timestep:)` inside the denoising loop. **Hot-path constraints**:
   - No `String` allocation per step.
   - No actor hop other than the (unavoidable) `await emit(...)`.
   - No call to `MemoryManager.shared.loadedComponentsMemory` per step.
   - Payload is three primitive `Int` values; the case constructor must be the only allocation.
3. Do **not** touch the LoRA branch or any `throw PipelineError` site — those belong to Sortie 6.
4. Confirm no accidental `try` slipped onto an `emit` call (the helper is non-throwing; a stray `try` would compile but signal a misunderstanding).

**Exit criteria**:
- [ ] `xcodebuild build -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` succeeds.
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO -only-testing:TuberiaTests` passes (no regressions in non-telemetry behavior).
- [ ] `grep -nE "\.generateStart\(|\.encodeStart\(|\.encodeComplete\(|\.decodeStart\(|\.decodeComplete\(|\.renderStart\(|\.renderComplete\(|\.generateComplete\(|\.denoiseStep\(" Sources/Tuberia/Pipeline/DiffusionPipeline.swift | wc -l` returns ≥ 9.
- [ ] `grep -n "MemoryManager.shared.loadedComponentsMemory" Sources/Tuberia/Pipeline/DiffusionPipeline.swift` shows no occurrence inside the denoising loop body (manual eyeball: confirm any match is outside the per-step loop).

---

## Sortie 6: DiffusionPipeline LoRA + errorThrown instrumentation

**Priority**: 12 — on the critical path. Completes pipeline instrumentation.

**Entry criteria**:
- [ ] Sortie 5 exit criteria checked.

**Tasks**:
1. In `Sources/Tuberia/Pipeline/DiffusionPipeline.swift`, wire the LoRA emit sites in the LoRA branch of `generate` (around lines 341–367):
   - `loraLoadStart` — before `LoRALoader.loadAdapterWeights` (line 346).
   - `loraMerge` — after adapter weights are loaded but before they're merged into the backbone (around lines 350–357).
   - `loraUnmerge` — after `LoRALoader.unapply` restores base weights at the end of `generate`.
2. Wire `errorThrown(phase: <call-site label>, error: <PipelineError>)` immediately before **every** `throw PipelineError.…` site reachable from public Tuberia API. The current file has 15 throw sites (lines 134, 144, 154, 169, 230, 327, 330, 333, 382, 400, 439, 463, 538, 554, 563); the count after Sortie 5's edits may differ slightly — verify by grep at sortie start.
3. Phase string convention per §5: dot-delimited, `<publicMethod>.<sub-step>[.<index>]`, e.g. `"loadModels.weightLoad"`, `"generate.encode"`, `"generate.decode"`, `"generate.render"`, `"generate.denoise.step.<i>"`. The `loadModels.memoryGate` site was already wired in Sortie 3 — do not duplicate it; verify it's still present.
4. Use the `if let telemetry { await telemetry.capture(.errorThrown(...)) }` pattern (not the `emit` helper) so `error` doesn't allocate when telemetry is off.
5. Confirm the throw still happens — the emit is a side-channel, not a swallow. Do not put the throw inside an `else` branch.

**Entry diagnostic command** (for the agent to run before editing):
```bash
grep -nE "throw PipelineError" Sources/Tuberia/Pipeline/DiffusionPipeline.swift
```
This produces the canonical list of sites the agent must instrument.

**Exit criteria**:
- [ ] `xcodebuild build -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` succeeds.
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO -only-testing:TuberiaTests` passes (no regressions).
- [ ] `THROWS=$(grep -c "throw PipelineError" Sources/Tuberia/Pipeline/DiffusionPipeline.swift); EMITS=$(grep -c "\.errorThrown(" Sources/Tuberia/Pipeline/DiffusionPipeline.swift); test "$EMITS" -ge "$THROWS"` succeeds (every throw site has at least one preceding errorThrown emit).
- [ ] `grep -nE "\.loraLoadStart\(|\.loraMerge\(|\.loraUnmerge\(" Sources/Tuberia/Pipeline/DiffusionPipeline.swift | wc -l` returns ≥ 3.

---

## Sortie 7: Generate / denoiseStep tests

**Priority**: 4.5 — verification for Sortie 5.

**Entry criteria**:
- [ ] Sortie 2 exit criteria checked.
- [ ] Sortie 5 exit criteria checked.

**Tasks**:
1. Create `Tests/TuberiaTests/Telemetry/GenerateTelemetryTests.swift` using Swift Testing.
2. Declare a `private struct GenerateTelemetryStubRecipe: PipelineRecipe, Sendable` inline (same in-target stub pattern as Sortie 4) with stubbed encoder, backbone, decoder, and renderer — no MLX work, no real weights, no GPU.
3. Assert `denoiseStep` event count exactly equals `request.steps`.
4. Assert canonical generate-flow ordering: `generateStart` → `encodeStart` → `encodeComplete` → (`denoiseStep` × steps) → `decodeStart` → `decodeComplete` → `renderStart` → `renderComplete` → `generateComplete`.
5. Assert `generateComplete.totalDurationSeconds` equals the value embedded in the returned `DiffusionGenerationResult.duration` (within float tolerance).

**Exit criteria**:
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO -only-testing:TuberiaTests/GenerateTelemetryTests` passes.
- [ ] `grep -n "MockPipelineRecipe" Tests/TuberiaTests/Telemetry/GenerateTelemetryTests.swift` returns no matches.
- [ ] No real MLX backbone / decoder is instantiated (verify by inspecting imports and stub initializers; no `MLX.` calls inside the recipe stub).

---

## Sortie 8: LoRA / errorThrown tests

**Priority**: 4.5 — verification for Sortie 6.

**Entry criteria**:
- [ ] Sortie 2 exit criteria checked.
- [ ] Sortie 6 exit criteria checked.

**Tasks**:
1. Create `Tests/TuberiaTests/Telemetry/LoRAErrorTelemetryTests.swift` using Swift Testing.
2. Declare an inline `private struct LoRAErrorStubRecipe: PipelineRecipe, Sendable` (same pattern).
3. Assert `loraLoadStart` / `loraMerge` / `loraUnmerge` fire **iff** `request.loRA != nil`. With `request.loRA == nil`, none of the three appear.
4. Stub the encoder to throw a known `PipelineError`; assert `errorThrown(phase: "generate.encode", …)` fires before the rethrown error reaches the caller. Verify the caller still observes the original error (not swallowed).
5. Repeat (4) for at least one decode-side throw (`phase: "generate.decode"`) and one render-side throw (`phase: "generate.render"`) using stubbed components.

**Exit criteria**:
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO -only-testing:TuberiaTests/LoRAErrorTelemetryTests` passes.
- [ ] `grep -nE "@Test" Tests/TuberiaTests/Telemetry/LoRAErrorTelemetryTests.swift | wc -l` returns ≥ 4 (LoRA-attached, LoRA-absent, encode-throw, plus decode or render throw).

---

## Sortie 9: WeightLoader + MemoryManager instrumentation

**Priority**: 13.5 — independent branch from the generate work; can run in parallel with Sortie 3 in the dispatch order, but plan layering puts it after Sortie 1.

**Entry criteria**:
- [ ] Sortie 1 exit criteria checked.
- [ ] Sortie 3 exit criteria checked (DiffusionPipeline now holds a telemetry reference, which this sortie will forward into `WeightLoader.load`).

**Tasks**:
1. In `Sources/Tuberia/Infrastructure/WeightLoader.swift`, add a defaulted `telemetry: (any TuberiaTelemetryReporter)? = nil` parameter (per §4.4 Option A) to `WeightLoader.load(...)` (line 33) and to `WeightLoader.loadFromPath(...)` (line 129) if it is part of the public API contract that consumers go through.
2. In `Sources/Tuberia/Pipeline/DiffusionPipeline.swift` `loadModels`, forward `self.telemetry` into each `WeightLoader.load(..., telemetry:)` invocation. Existing direct callers of `WeightLoader.load` in downstream packages (`pixart-swift-mlx`, `flux-2-swift-mlx`) compile unchanged because the parameter defaults to `nil`.
3. In `Sources/Tuberia/Infrastructure/MemoryManager.swift`, add `var telemetry: (any TuberiaTelemetryReporter)? = nil` and `public func setTelemetry(_ reporter: (any TuberiaTelemetryReporter)?) { self.telemetry = reporter }` per §4.5.
4. Wire `MemoryManager.clearGPUCache` (line 119) to emit `unloadSegment(role: nil, componentId: nil, metalAllocatedBytesBefore: <metal>, metalAllocatedBytesAfter: <metal>)` around the existing `MLX.Memory.clearCache()` call. Cache `MTLCreateSystemDefaultDevice()` in a stored `let` (instance- or actor-isolated, computed lazily on first use) and read `?.currentAllocatedSize` on it to populate the Metal bytes.
5. Confirm `clearGPUCache` is the only emitter of the `role: nil` `unloadSegment` apart from any future call sites — `DiffusionPipeline.unloadModels` should NOT itself emit a terminal `role: nil` event, since it already calls `clearGPUCache` and the cache-clear emit covers that probe.

**Exit criteria**:
- [ ] `xcodebuild build -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` succeeds.
- [ ] `grep -n "telemetry: (any TuberiaTelemetryReporter)? = nil" Sources/Tuberia/Infrastructure/WeightLoader.swift` returns ≥ 1 line.
- [ ] `grep -nE "WeightLoader\.load\(" Sources/Tuberia/Pipeline/DiffusionPipeline.swift` shows the call sites pass `telemetry: self.telemetry` (or equivalent forwarding).
- [ ] `grep -n "setTelemetry" Sources/Tuberia/Infrastructure/MemoryManager.swift` returns the new public setter.
- [ ] `grep -nE "\.unloadSegment\(role: nil" Sources/Tuberia/` returns exactly **1** match, inside `MemoryManager.clearGPUCache`.

---

## Sortie 10: WeightLoader + MemoryManager tests

**Priority**: 4.5 — verification for Sortie 9.

**Entry criteria**:
- [ ] Sortie 2 exit criteria checked.
- [ ] Sortie 9 exit criteria checked.

**Tasks**:
1. Create `Tests/TuberiaTests/Telemetry/WeightLoaderTelemetryTests.swift` reusing the synthetic safetensors fixture pattern from `Tests/TuberiaTests/MemoryManagerTests.swift` and any existing `WeightLoaderIntegrationTests` pattern (no real model weights, no GPU).
2. Assert: defaulted (`nil`) telemetry parameter produces zero events while `WeightLoader.load` still succeeds on the synthetic fixture.
3. Assert: with a `MockTuberiaTelemetryReporter` set, `segmentWeightsLoaded` fires with `safetensorsFileCount > 0` and `parameterCount > 0`. (This event is emitted by the *pipeline*, not WeightLoader; if the test invokes `WeightLoader.load` directly, scope the assertion to the events the new `telemetry:` parameter forwards. If WeightLoader emits no events itself by design, document that fact in the test as a comment and assert event count == 0 with reporter attached.)
4. Create `Tests/TuberiaTests/Telemetry/MemoryManagerTelemetryTests.swift` using Swift Testing.
5. Assert: `MemoryManager.shared.clearGPUCache()` with a reporter set emits exactly one `unloadSegment(role: nil, componentId: nil, …)` event whose `metalAllocatedBytesBefore` and `metalAllocatedBytesAfter` are non-nil.
6. Skip the Metal-bytes check (with a clear `#expect(throws: ...)` or `try #require(MTLCreateSystemDefaultDevice() != nil)`) when no Metal device is available so the suite is portable.

**Exit criteria**:
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO -only-testing:TuberiaTests/WeightLoaderTelemetryTests -only-testing:TuberiaTests/MemoryManagerTelemetryTests` passes.
- [ ] No GPU work runs in `WeightLoaderTelemetryTests` (synthetic safetensors only — verify by absence of `MLX.` calls in the test file).

---

## Sortie 11: Performance regression suite + Makefile

**Priority**: 5.5 — gates release readiness.

**Entry criteria**:
- [ ] Sortie 3 exit criteria checked.
- [ ] Sortie 5 exit criteria checked.
- [ ] Sortie 6 exit criteria checked.
- [ ] Sortie 9 exit criteria checked.

**Tasks**:
1. Create `Tests/TuberiaTests/Telemetry/TelemetryOverheadTests.swift` using Swift Testing. The suite is gated behind a Makefile target — it must NOT run in the default `make test` or in CI.
2. Test runs an in-target stubbed pipeline end-to-end (load + generate + unload) twice: once with `nil` reporter, once with `NoopTuberiaTelemetryReporter()`. 10 trials each, alternating order to dampen warm-up bias.
3. Assertion threshold: wall-clock delta **< 10%** between the two configurations. (REQUIREMENTS §6 sets a 1% target; §8.2 acknowledges "the goal is to catch a 10% regression, not a 0.5% one — Swift Testing parallel scheduling on macos-26 is noisy". The codified test threshold is 10%; the 1% number is a project goal, not the assertion threshold.)
4. Add a dedicated `make test-telemetry-overhead` target to the existing `Makefile`. The target invokes:
   ```
   xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO -only-testing:TuberiaTests/TelemetryOverheadTests
   ```
   Add a comment in the Makefile target stating it is invoked manually before each release and is intentionally excluded from `make test`.
5. Tag the test with a `.disabled` trait or a `@Suite(.serialized)` container plus an env-var gate (e.g. `TUBERIA_TELEMETRY_OVERHEAD=1`) so a stray invocation of `make test` does not pick it up. The `-only-testing` flag in (4) is the primary gate; the env-var/trait is belt-and-suspenders.

**Exit criteria**:
- [ ] `make test-telemetry-overhead` runs the new suite and reports a delta.
- [ ] `make test` does NOT include `TelemetryOverheadTests` — verify by running `make test` and confirming the suite name is absent from the report.
- [ ] `grep -n "test-telemetry-overhead:" Makefile` returns 1 line.
- [ ] The Makefile target's xcodebuild invocation includes `-parallel-testing-enabled NO`: `grep -A3 "test-telemetry-overhead:" Makefile | grep -c "parallel-testing-enabled NO"` returns ≥ 1.

---

## Sortie 12: Documentation

**Priority**: 4.5 — sub-agent candidate (no Swift compilation, pure markdown).

**Sub-agent eligibility**: Yes. This sortie touches only `AGENTS.md` and `README.md`; no Swift code, no `xcodebuild`. Can be dispatched to a sub-agent in parallel with sorties that share its dependencies.

**Entry criteria**:
- [ ] Sortie 1 exit criteria checked (public type names are stable).
- [ ] Sortie 3 exit criteria checked (`setTelemetry` exists; the README snippet must compile against the live API).

**Tasks**:
1. Update `AGENTS.md`: add a "Telemetry" section pointing at `Sources/Tuberia/Telemetry/`, explaining the opt-in contract (nil reporter ⇒ zero events) and listing the major event categories (lifecycle, generation, LoRA, errors).
2. Append a "How Produciesta consumes this" appendix in `AGENTS.md` linking to `../Produciesta/MULTI_REPO_TELEMETRY.md`. Informational only — must NOT introduce a Swift-level dependency.
3. Update `README.md` with the consumer snippet from REQUIREMENTS §9 (verbatim or near-verbatim):
   ```swift
   let pipeline = try DiffusionPipeline(recipe: recipe)
   await pipeline.setTelemetry(MyReporter())   // optional
   try await pipeline.loadModels(progress: { _, _ in })
   ```
4. Reference the test-side example reporter `Tests/TuberiaTests/Telemetry/MockTuberiaTelemetryReporter.swift` from the README so consumers have a concrete starting point.
5. Sanity-check that no documentation file implies a runtime dependency on Produciesta — narrative references are fine; `import Produciesta` in code blocks is not.

**Exit criteria**:
- [ ] `grep -n "## Telemetry" AGENTS.md` returns ≥ 1 line.
- [ ] `grep -n "How Produciesta consumes this" AGENTS.md` returns ≥ 1 line.
- [ ] `grep -n "setTelemetry\|TuberiaTelemetryReporter" README.md` returns ≥ 1 line each.
- [ ] `grep -rn "import Produciesta" Sources/` returns no matches (sanity).
- [ ] The README snippet uses identifiers that exist in `Sources/Tuberia/`: `grep -n "DiffusionPipeline\|setTelemetry\|loadModels" Sources/Tuberia/Pipeline/DiffusionPipeline.swift` returns matches for each.

---

## Sortie 13: Acceptance verification

**Priority**: 1.5 — terminal checkpoint.

**Entry criteria**:
- [ ] Sorties 1–12 all `COMPLETED`.

**Tasks**:
1. Run the full test command from REQUIREMENTS §12 verbatim:
   ```
   xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO
   ```
   Confirm it includes the new `Tests/TuberiaTests/Telemetry/` suite and passes.
2. `git diff origin/main -- Package.swift` and confirm zero net-new dependencies. (If `origin/main` is unavailable, diff against the mission's recorded `starting_point_commit` from SUPERVISOR_STATE.md.)
3. `grep -r "import Produciesta\|import ProduciestaCore" Sources/Tuberia/` — must return nothing.
4. Walk every check in REQUIREMENTS §12 and tick the box. Produce `MISSION_ACCEPTANCE.md` at the project root listing each §12 item with the exact verifying command and its output (or a one-line summary of the output).
5. Confirm the work ships as our next minor release version (additive public API only; no breaking changes). Do **not** edit `Package.swift` to set a version number — version selection happens at release time via the `release` skill.

**Exit criteria**:
- [ ] Full `xcodebuild test … -parallel-testing-enabled NO` passes.
- [ ] `Package.swift` dependency list unchanged in net.
- [ ] No `Produciesta` / `ProduciestaCore` imports in `Sources/Tuberia/`.
- [ ] `MISSION_ACCEPTANCE.md` exists at project root, listing all REQUIREMENTS §12 items as ticked with verifying commands.
- [ ] `Sources/Tuberia/Telemetry/{TuberiaTelemetryEvent,TuberiaTelemetryReporter,NoopTuberiaTelemetryReporter}.swift` all exist and are exported from the `Tuberia` product.

---

## Summary

| Metric                | Value                                                           |
|-----------------------|-----------------------------------------------------------------|
| Work units            | 1                                                               |
| Total sorties         | 13 (was 11; Sortie 5 split into 5+6, Sortie 6 split into 7+8)   |
| Critical path length  | 6 sorties (1 → 3 → 5 → 6 → 11 → 13)                            |
| Parallel groups       | 6 (A–F); virtually all sorties supervising-agent only           |
| Sub-agent candidates  | 1 (Sortie 12 — documentation, no build)                         |
| Source                | `REQUIREMENTS-telemetry.md`                                     |
| Public-API impact     | additive (new types + setters + defaulted parameters)           |
| Version target        | our next minor release version                                  |

## Open Questions & Missing Documentation

After all 4 refinement passes, **no blocking open questions remain**. All four open questions inherited from REQUIREMENTS §13 are locked under "Decisions Locked from Source §13" above. The MockPipelineRecipe target-visibility gap is resolved by the inline-stub-recipe pattern documented in "Project Conventions Locked" and applied in Sorties 4, 7, 8.

Items the supervisor will revisit if they surface during execution:

- The post-cache-clear `unloadSegment(role: nil, …)` test (Sortie 10 task 5) requires a real Metal device; on a host without one the test should `try #require(MTLCreateSystemDefaultDevice() != nil)` and be skipped. CI on `macos-26` always has a Metal device, so this is a portability nicety, not a blocker.
- If `WeightLoader.load` does not itself emit any events (the requirements suggest emits happen at the pipeline boundary, not inside the loader), Sortie 10 task 3 collapses to "assert event count == 0 from the loader-direct path". The agent should make this call from the source of truth in Sortie 9 and document the choice in a comment.
