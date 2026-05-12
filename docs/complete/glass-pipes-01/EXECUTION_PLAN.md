---
mission: swift-tuberia-instrumentation
feature_name: OPERATION GLASS PIPES
source_requirements: REQUIREMENTS-instrumentation.md
parent_plan: ../Vinetas/EXECUTION_PLAN.md
scope: single-repo (SwiftTuberia)
critical_path: true
publishes: TuberiaTensorStat (consumed by flux-2-swift-mlx, pixart-swift-mlx, SwiftVinetas)
target_branch: instrumentation/01
mission_branch: instrumentation/01
starting_point_commit: b6f3ba6414061d413b29b27cf6711605afebcef5
iteration: 1
release_type: minor
refined: true
refine_passes: [atomicity, priority, parallelism, questions]
state: completed
---

# EXECUTION_PLAN.md — SwiftTuberia Instrumentation

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase). This plan defines a single work unit: SwiftTuberia.

---

## Mission Overview

Goal: instrument SwiftTuberia's five protocol seams (`TextEncoder → Scheduler → Backbone → Decoder → Renderer`) with a telemetry surface that:

1. Publishes the shared `TuberiaTensorStat` Codable struct that downstream MLX libraries depend on.
2. Emits boundary events for assembly validation, memory gating, weight load, LoRA merge/unmerge, text encoder handoff, scheduler config, per-step denoise, the CFG dtype cast, backbone/decoder/renderer handoffs, numerical anomalies, and thrown errors — per §5 of `REQUIREMENTS-instrumentation.md`.
3. Costs zero when telemetry is nil. The denoise loop is the inner loop of the library; every hot-path tensor sample is gated by `if let telemetry { ... }` so `TuberiaTensorStat.sample(...)` (8 MLX reductions per call) never runs when telemetry is off.
4. Is shipped as a minor version bump so `flux-2-swift-mlx`, `pixart-swift-mlx`, and `SwiftVinetas` can pin against it.

**Source of truth:** `/Users/stovak/Projects/SwiftTuberia/REQUIREMENTS-instrumentation.md` — §3 (public types), §4 (injection points), §5 (per-event emission), §7 (tests). §1, §2, §8 are context. §6 (Vinetas adapter mapping) is host-side and OUT OF SCOPE here.

### Critical-path note

This is the most upstream work unit in the cross-repo Vinetas instrumentation campaign. The `TuberiaTensorStat` Codable struct published by Sortie 1 is imported by `flux-2-swift-mlx`, `pixart-swift-mlx`, and `SwiftVinetas` — none of those repos can compile their instrumentation slices until Sortie 1 lands and a tagged release exists.

### Repo-specific execution rules

- All sorties land on a single PR branch `instrumentation/01` against SwiftTuberia's default branch.
- Builds use **XcodeBuildMCP** locally (`swift_package_build`, `swift_package_test`). **Never** use raw `swift build` / `swift test`.
- Prefer the `make build` / `make test` / `make lint` targets (Makefile is in repo root).
- After all sorties complete and the PR merges, tag the repo with our next minor release version so downstream libs can pin against it. No concrete version number — release tooling resolves this at tag time.

---

## Open Questions & Missing Documentation

These items were surfaced by Pass 4 and should be resolved (or explicit decisions recorded in the PR description) before or during the relevant sortie.

| # | Sortie | Issue Type | Description | Recommendation |
|---|--------|-----------|-------------|----------------|
| Q1 | Sortie 5 | Open question | The parent task references a `protocolHandoff` event but `REQUIREMENTS-instrumentation.md` §3.2 does **not** define one. The events that correspond to "protocol handoff" are `textEncoderForward{Start,Complete}`, `backboneForward{Start,Complete}`, `decoderDecode{Start,Complete}`, `rendererRender{Start,Complete}` — eight events across four seams, each carrying a `TuberiaTensorStat` for the boundary tensor. | **RESOLVED IN PLAN**: treat the four start/complete pairs as the canonical handoff events. The autoclosure-discipline rule applies to every call site that constructs a `TuberiaTensorStat`. If the parent campaign needs an umbrella `protocolHandoff` event later, it is additive and out of scope here. |
| Q2 | Sortie 1 | Vague criterion | Requirements §3.2 mentions `AnomalyKind.outOfRange` with "default >1e6 in magnitude" but does not say where the threshold is configured. | **RESOLUTION**: hard-code the default at `1e6` in `TuberiaTensorStat` (or in the call-site emission of `numericalAnomaly`). If configurability is later needed, add it as a public static var on `TuberiaTensorStat`. Document the chosen location in the PR description. |
| Q3 | Sortie 3, 4, 5 | External dependency | The emission table in §5 references specific line numbers (e.g. `:511`, `:545`, `:480`) in `DiffusionPipeline.swift`. These may have drifted since the requirements doc was written. | **RESOLUTION**: each sortie's first action is to `grep` for the anchor pattern (e.g. `.asType(.float32)` for line 511, `scheduler.configure` for line 475) and use the current line. Document any drift in PR description. |
| Q4 | Sortie 5 | Open question | The `numericalAnomaly` side-channel is described in §5 as firing "from inside any emission that sampled a tensor" — but `TuberiaTensorStat.sample(...)` has no telemetry reference. | **RESOLUTION IN PLAN**: keep `sample()` pure (no side effects). Emit `numericalAnomaly` from the call sites after inspecting the returned stat. Centralize via a private helper in `DiffusionPipeline` so the inspect-and-emit pattern is one-liner at the use site. Recorded in Sortie 5 task 7. |
| Q5 | Sortie 6 | Vague criterion | Requirements §7 row 5 says "Both wall-clock medians within ±2% over 30 iterations" but the parent mission constraint is "+1% on a 50-step denoise loop." | **RESOLUTION IN PLAN**: use the stricter constraint (±1% over 30 iterations on a 50-step loop) for Sortie 7. Document deviation from §7 in the PR description. |

No Pass 4 issue is currently classified as BLOCKING — Q1 through Q5 are all resolved in-plan with recorded recommendations. The supervisor may proceed to execution.

---

## Work Units

| # | Work Unit | Directory | Sorties | Layer | Dependencies |
|---|-----------|-----------|---------|-------|--------------|
| WU1 | swift-tuberia-instrumentation | `/Users/stovak/Projects/SwiftTuberia` | 7 | 0 | none (this repo is upstream of flux/pixart/SwiftVinetas) |

Single-work-unit plan. All sorties execute sequentially on branch `instrumentation/01`.

---

## Parallelism Structure

**Critical Path**: Sortie 1 → 2 → 3 → 4 → 5 → 6 → 7 (length: 7 sorties — strictly linear).

**Parallel Execution Groups**:

| Group | Sorties | Notes |
|-------|---------|-------|
| (single) | All 7 | All sorties touch overlapping files (`DiffusionPipeline.swift` at minimum) and depend on the previous sortie's compile-clean state. No intra-WU parallelism is safe. |

**Inter-repo parallelism opportunity** (handled by parent plan `/Users/stovak/Projects/Vinetas/EXECUTION_PLAN.md`, not by this plan): once **Sortie 1 merges and the minor-version tag is cut**, `flux-2-swift-mlx`, `pixart-swift-mlx`, and `SwiftVinetas` can begin their instrumentation slices in parallel. Until then, those repos are blocked on this work unit's Sortie 1.

**Agent Constraints**:
- **Supervising agent**: every sortie (all 7) has build/test/lint steps and must be dispatched to the supervising agent. No sub-agent allocation in this plan.
- **Sub-agents**: 0 (none — no build-free work in this plan).

**Maximum parallelism within this plan**: 1.

---

## WU1 — SwiftTuberia Instrumentation

**Goal:** land all of §3 (types), §4 (injection points), §5 (emission), §7 (tests) on `instrumentation/01` and prove that telemetry-off overhead stays within +1% of un-instrumented baseline on a 50-step denoise loop.

---

### Sortie 1: Publish `TuberiaTensorStat` and the telemetry surface types

**Priority**: 21 — Highest. This sortie publishes the cross-repo type. flux-2-swift-mlx, pixart-swift-mlx, and SwiftVinetas are all blocked on its release tag. Dependency depth 5, foundation 1, risk 3 (new public API with MLX reductions), complexity 2.

**Entry criteria:**
- [ ] First sortie — no prerequisites.
- [ ] Branch `instrumentation/01` checked out from the default branch HEAD.
- [ ] `make build` succeeds on the unmodified checkout (baseline green).

**Tasks:**
1. Create `Sources/Tuberia/Telemetry/` directory.
2. Implement `Sources/Tuberia/Telemetry/TuberiaTensorStat.swift` per §3.1 — `public struct`, `Codable`, `Sendable`, with a real `static func sample(_ array: MLXArray) -> TuberiaTensorStat` that:
   - Captures shape and dtype string (canonical MLX names: `float16`, `float32`, `bfloat16`, `int32`, etc.).
   - Computes min/max/mean/std as MLX reductions.
   - Computes `hasNaN` via `.isNaN().any()` and `hasInf` via `.isInf().any()`.
   - Calls `eval()` exactly once on the tuple of reductions.
   - Casts all numeric outputs to `Double` so float16/32/bfloat16 inputs produce the same struct shape.
   - **Does NOT emit telemetry side-effects** (per Q4 in Open Questions — `sample()` is pure).
3. Implement `Sources/Tuberia/Telemetry/TuberiaTelemetryEvent.swift` per §3.2 — full `public enum` with all 22 cases and the 5 nested enums (`AssemblyCheck`, `TextEncoderRole`, `BackboneBranch`, `AnomalyKind`, `ErrorPhase`).
4. Implement `Sources/Tuberia/Telemetry/TuberiaTelemetryReporter.swift` per §3.3 — `public protocol TuberiaTelemetryReporter: Sendable { func capture(_ event: TuberiaTelemetryEvent) async }` + `public struct NoopTuberiaTelemetryReporter: TuberiaTelemetryReporter`.
5. Add `Tests/TuberiaTests/TuberiaTensorStatPublicAPITests.swift` — imports `Tuberia`, instantiates `TuberiaTensorStat(shape: [1], dtype: "float32", min: 0, max: 0, mean: 0, std: 0, hasNaN: false, hasInf: false)`, and JSON-encodes + decodes it through `JSONEncoder`/`JSONDecoder`. Asserts round-trip equality on every field. This proves the type is `public`, `Codable`, and `Sendable`.

**Exit criteria:**
- [ ] `make build` succeeds.
- [ ] `make test` runs the new `TuberiaTensorStatPublicAPITests` and it passes.
- [ ] `swift_package_build` via XcodeBuildMCP succeeds.
- [ ] All three new files exist under `Sources/Tuberia/Telemetry/` (`TuberiaTensorStat.swift`, `TuberiaTelemetryEvent.swift`, `TuberiaTelemetryReporter.swift`).
- [ ] `grep -r "fatalError" Sources/Tuberia/Telemetry/` returns no matches (no stubs remain in `sample()`).
- [ ] `grep -E "^public " Sources/Tuberia/Telemetry/TuberiaTensorStat.swift` shows `struct TuberiaTensorStat`, its initializer, and the `static func sample` are all `public`.
- [ ] `grep -E "^public " Sources/Tuberia/Telemetry/TuberiaTelemetryReporter.swift` shows the protocol and Noop struct are both `public`.

---

### Sortie 2: Telemetry injection points (setters and defaulted parameters)

**Priority**: 15.5 — High. Plumbing-only; no emission logic. Dependency depth 4, foundation 1, risk 1, complexity 1.

**Entry criteria:**
- [ ] Sortie 1 exit criteria satisfied.

**Tasks:**
1. Add `setTelemetry(_:)` extension method on `DiffusionPipeline` per §4.1 in a new file `Sources/Tuberia/Pipeline/DiffusionPipeline+Telemetry.swift`. Method signature: `public func setTelemetry(_ reporter: (any TuberiaTelemetryReporter)?)`.
2. Add private ivar `private var telemetry: (any TuberiaTelemetryReporter)? = nil` to the `DiffusionPipeline` actor (in the main `DiffusionPipeline.swift` file).
3. Add defaulted `telemetry: (any TuberiaTelemetryReporter)? = nil` parameter to:
   - `LoRALoader.loadAdapterWeights(config:keyMapping:)` per §4.2
   - `LoRALoader.apply(adapterWeights:to:scale:)` per §4.2
   - `WeightLoader.load(componentId:keyMapping:)` per §4.3
   - `MemoryManager.hardValidate(requiredBytes:)` per §4.4
4. Update `DiffusionPipeline.memoryGate` closure default so it forwards `self.telemetry` to `MemoryManager.hardValidate`.
5. **No emission logic yet** — only plumbing. Each new parameter is referenced as `_ = telemetry` (or left unused if the compiler does not warn) until Sortie 3+ wires it up.

**Exit criteria:**
- [ ] `make build` succeeds.
- [ ] `make test` passes (defaulted parameters preserve source compatibility — existing call sites still compile).
- [ ] `make lint` shows no new warnings introduced by this sortie.
- [ ] `grep -E "telemetry: \\(any TuberiaTelemetryReporter\\)\\?" Sources/Tuberia/Pipeline/LoRALoader.swift Sources/Tuberia/Infrastructure/WeightLoader.swift Sources/Tuberia/Infrastructure/MemoryManager.swift` shows all four call sites carry the defaulted parameter.
- [ ] `grep -n "func setTelemetry" Sources/Tuberia/Pipeline/DiffusionPipeline+Telemetry.swift` returns exactly one match.
- [ ] `grep -n "private var telemetry" Sources/Tuberia/Pipeline/DiffusionPipeline.swift` returns exactly one match.

---

### Sortie 3: Lifecycle, assembly, memory, weight, and LoRA emission

**Priority**: 12.25 — Medium-high. Five emission groups, none on the hot path. Dependency depth 3, foundation 0, risk 2, complexity 2.5. Borderline-large but coherent (all use the `if let telemetry { ... }` template, none in a tight loop).

**Entry criteria:**
- [ ] Sortie 2 exit criteria satisfied.

**Tasks:**
1. **Anchor verification (per Q3):** before touching `DiffusionPipeline.swift`, grep for `validateAssembly`, `loadModels`, `LoRALoader.loadAdapterWeights`, and record the current line numbers in PR description. Use those, not the §5 numbers, if drift is detected.
2. Emit `pipelineConfigured` at end of `DiffusionPipeline.init(recipe:)` per §5.
3. Emit `pipelineStart` at `generate(...)` entry. Emit `pipelineEnd` on both success (carrying `success: true`) and error paths (via `defer` that reads a local `var success = false; … success = true` pattern; carry `success: false` on the error path).
4. Emit `assemblyCheckPassed` after each of the six checks in `validateAssembly` (`completeness`, `encoderToBackboneDim`, `encoderToBackboneSeq`, `backboneToDecoder`, `decoderToRenderer`, `imageToImageBidirectional`).
5. Emit `assemblyCheckFailed` immediately before each corresponding `throw PipelineError.incompatibleComponents(...)` block. Reuse the inlet/outlet/reason already built for the error.
6. Emit `memoryGateChecked` after `try await memoryGate(peak)`.
7. Emit `weightLoadStart` / `weightLoadComplete` pairs around each `WeightLoader.load` invocation. Compute `paramCount` and `totalBytes` from the returned `ModuleParameters`.
8. Emit `loraLoadStart` / `loraLoadComplete` pairs around `LoRALoader.loadAdapterWeights`. Emit `loraApplied` after `backbone.apply(weights: mergedWeights)`. Emit `loraUnapplied` after the deferred unmerge.
9. Emit `componentReadinessChecked` inside `loadModels` per component.
10. Add `errorThrown` emission immediately before every `throw` in: assembly (lines ~134, 144, 154, 169), loadModels (~227, 230), missing-component (~327, 330, 333), and `LoRALoader.swift:44`, `WeightLoader.swift:51, 60, 98, 100, 102` (subject to Q3 anchor verification).
11. **Discipline rule:** every emission site uses the `if let telemetry { ... }` template. No `@autoclosure` on the protocol — guard at each call site.

**Exit criteria:**
- [ ] `make build` succeeds.
- [ ] `make test` passes (existing tests must remain green; new tests come in Sortie 6).
- [ ] `grep -cE "throw " Sources/Tuberia/Pipeline/DiffusionPipeline.swift Sources/Tuberia/Pipeline/LoRALoader.swift Sources/Tuberia/Infrastructure/WeightLoader.swift` count matches `grep -cE "errorThrown\\(" Sources/Tuberia/Pipeline/DiffusionPipeline.swift Sources/Tuberia/Pipeline/LoRALoader.swift Sources/Tuberia/Infrastructure/WeightLoader.swift` on the lines in scope for this sortie (assembly + loadModels + LoRA/Weight loaders). Generation/encoder/decoder throws stay unhandled until Sorties 4, 5.
- [ ] `grep -n "TuberiaTensorStat.sample" Sources/Tuberia/Pipeline/DiffusionPipeline.swift` — every match must be inside an `if let telemetry` block (manual review for now; Sortie 5 adds the automated grep count check).
- [ ] PR description records any line-number drift discovered during task 1.

---

### Sortie 4: Text-encoder and scheduler emission

**Priority**: 7.5 — Medium. Two emission groups, no hot path. Dependency depth 2, foundation 0, risk 1, complexity 1.

**Entry criteria:**
- [ ] Sortie 3 exit criteria satisfied.

**Tasks:**
1. **Anchor verification (per Q3):** grep for `encoder.encode`, `scheduler.configure` in `DiffusionPipeline.swift` and confirm line numbers.
2. Emit `textEncoderForwardStart` / `textEncoderForwardComplete` pairs around `encoder.encode(encoderInput)` (~`:380`) with `role: .conditional`, and around `encoder.encode(uncondInput)` (~`:398`) with `role: .unconditional`. Sample `embeddingStat` and `maskStat` inside the `if let telemetry { ... }` guard.
3. Emit `schedulerConfigured` after `scheduler.configure(...)` returns (~`:475`). Compute `timestepsHead` = first 5 of `plan.timesteps`, `timestepsTail` = last 5; same for `sigmasHead`/`sigmasTail`. Carry full `predictionType` string from the scheduler.
4. Add `errorThrown` emits before `throw` at lines ~382, ~400 (encoding throws) and any scheduler-configure throw site.

**Exit criteria:**
- [ ] `make build` succeeds.
- [ ] `make test` passes.
- [ ] `grep -nE "case (textEncoderForwardStart|textEncoderForwardComplete|schedulerConfigured)" Sources/Tuberia/Pipeline/DiffusionPipeline.swift` returns at least 5 matches (2 + 2 + 1).
- [ ] Every `TuberiaTensorStat.sample(` call added by this sortie is inside an `if let telemetry { ... }` block (manual review).

---

### Sortie 5: Hot-path emission — denoise loop, CFG cast, backbone/decoder/renderer

**Priority**: 7.25 — Medium-high (risk-dominated). This is the cost-critical sortie. Dependency depth 1, foundation 0, risk 3 (the entire +1% overhead bar depends on getting the guards right here), complexity 2.5.

**Entry criteria:**
- [ ] Sortie 4 exit criteria satisfied.

**Tasks:**
1. **Anchor verification (per Q3):** grep for the loop header (~`:480`), `.asType(.float32)` (~`:511`, ~`:529`), `eval(latents)` (~`:545`), `backbone.forward` (~`:502, :503, :529`), `decoder.decode` (~`:552`), `renderer.render` (~`:560`). Record current line numbers in PR description.
2. Emit `denoiseStepStart` at the top of each iteration (~`:480`). Sample `latentBeforeStat` inside the guard.
3. Emit `denoiseStepComplete` after `eval(latents)` (~`:545`). Sample `latentAfterStat` and `predictionStat` inside the guard.
4. Emit `cfgDtypeCast` immediately after each `.asType(.float32)` cast (~`:511` CFG branch, ~`:529` non-CFG branch). Sample `guidedPredictionStat` post-cast.
5. Emit `backboneForwardStart` / `backboneForwardComplete` around each `backbone.forward(...)` call. Use `BackboneBranch.cfgConditional` (~`:502`), `BackboneBranch.cfgUnconditional` (~`:503`), `BackboneBranch.noCFG` (~`:529`).
6. Emit `decoderDecodeStart` / `decoderDecodeComplete` around `decoder.decode(latents)` (~`:552`). Carry `scalingFactor`.
7. Emit `rendererRenderStart` / `rendererRenderComplete` around `renderer.render(...)` (~`:560`). Carry rendered `outputBytes`.
8. **Numerical anomaly side-channel (per Q4)**: in each `if let telemetry` block that calls `TuberiaTensorStat.sample(...)`, after constructing the stat, also call a private helper `emitAnomalyIfPresent(stat:phase:stepIndex:)` that checks `stat.hasNaN || stat.hasInf || abs(stat.max) > 1e6 || abs(stat.min) > 1e6` and emits `numericalAnomaly` with the appropriate `AnomalyKind`. Place the helper in `DiffusionPipeline+Telemetry.swift`. Default threshold `1e6` (per Q2).
9. Add `errorThrown` emits before `throw` at lines ~439, ~463, ~538 (generation), ~554 (decoding), ~563 (rendering).
10. **Hot-path discipline:** every stat sample is inside `if let telemetry { ... }`. Six tensors are sampled per CFG step (`latentBefore`, `latentAfter`, `prediction`, `guidedPrediction`, plus two backbone-branch inputs). 8 reductions × 6 stats = 48 reductions/step in CFG path; with telemetry nil, exactly **zero** reductions execute — this is what Sortie 7 measures.

**Exit criteria:**
- [ ] `make build` succeeds.
- [ ] `make test` passes (existing tests still green).
- [ ] `grep -cn "TuberiaTensorStat.sample(" Sources/Tuberia/Pipeline/DiffusionPipeline.swift` == count of `if let telemetry` blocks that contain a `sample(` call (the verifier writes both counts to the PR description; equality is required).
- [ ] `grep -c "case denoiseStepStart\\|case denoiseStepComplete\\|case cfgDtypeCast\\|case backboneForward\\|case decoderDecode\\|case rendererRender\\|case numericalAnomaly" Sources/Tuberia/Pipeline/DiffusionPipeline.swift` returns at least 11 (5 hot-path events + 2 boundary events + anomaly side-channel + 3 backbone-branch emissions).
- [ ] All rows 14–22 of §5's emission table are wired (manual checklist in PR description).
- [ ] PR description records line-number drift findings from task 1.

---

### Sortie 6: Functional tests (assembly, denoise, CFG cast, anomaly, LoRA)

**Priority**: 6 — Medium. Five functional test files. Dependency depth 1, foundation 0, risk 2, complexity 2.

**Entry criteria:**
- [ ] Sortie 5 exit criteria satisfied.

**Tasks:**
1. Add `Tests/TuberiaTests/TuberiaTelemetryAssemblyTests.swift` per §7 row 1 — two test cases: (a) deliberately-mismatched-recipe failure path asserts `assemblyCheckFailed(.encoderToBackboneDim, ...)` fires **before** the `PipelineError` is thrown; (b) valid-recipe path asserts all six `assemblyCheckPassed` events fire in order.
2. Add `Tests/TuberiaTests/TuberiaTelemetryDenoiseLoopTests.swift` per §7 row 2 — `MockBackbone` and `MockScheduler` returning deterministic tensors, run 4 steps. Assert: 4× `denoiseStepStart`, 4× `denoiseStepComplete`, `stepIndex` monotone, `latentAfterStat.shape` matches configured latent shape.
3. Add `Tests/TuberiaTests/TuberiaTelemetryCFGCastTests.swift` per §7 row 3 — drive a CFG run; assert `cfgDtypeCast(fromDtype: "float16", toDtype: "float32", ...)` fires per step and `guidedPredictionStat.dtype == "float32"`.
4. Add `Tests/TuberiaTests/TuberiaTelemetryAnomalyTests.swift` per §7 row 4 — `MockBackbone` injects NaN at step 2. Assert: (a) `denoiseStepComplete` for step 2 carries `predictionStat.hasNaN == true`; (b) `numericalAnomaly(phase: "backbone_forward_complete", kind: .nan, stepIndex: 2, ...)` is emitted within the same step boundary.
5. Add `Tests/TuberiaTests/TuberiaTelemetryLoRATests.swift` per §7 row 6 — run a generation with a LoRA config; assert `loraLoadStart`, `loraLoadComplete`, `loraApplied`, `loraUnapplied` fire in correct order.
6. All five test files use a `RecordingTelemetryReporter` test fixture (collects `[TuberiaTelemetryEvent]` in an actor) which should be placed in `Tests/TuberiaTests/Support/RecordingTelemetryReporter.swift`.

**Exit criteria:**
- [ ] `make build` succeeds.
- [ ] `make test` passes — all five new test files green.
- [ ] `make lint` clean.
- [ ] `find Tests/TuberiaTests -name "TuberiaTelemetry*.swift" -not -name "*Overhead*"` returns exactly 5 files.
- [ ] `find Tests/TuberiaTests/Support -name "RecordingTelemetryReporter.swift"` returns 1 file.

---

### Sortie 7: Noop overhead measurement + PR open

**Priority**: 3.75 — Risk-dominated. The +1% overhead bar is the highest-risk single check in the entire mission. Dependency depth 0, foundation 0, risk 3, complexity 1.5.

**Entry criteria:**
- [ ] Sortie 6 exit criteria satisfied.

**Tasks:**
1. Add `Tests/TuberiaTests/TuberiaTelemetryNoopOverheadTests.swift` per §7 row 5, **adapted** per Q5: use a **50-step** denoise loop (not 4) with deterministic mocks. Compare `nil` reporter vs. `NoopTuberiaTelemetryReporter`. Run 30 iterations of each. Compute medians.
2. Assertion: median wall-clock delta ≤ **+1.0%** (the parent-mission bar; stricter than §7's ±2%).
3. Test must be marked appropriately for CI vs. local. If the test is too noisy on shared CI hardware, gate it behind an environment variable `TUBERIA_OVERHEAD_TEST=1` and document the local-run command in the PR description.
4. Run the test locally via `make test` (or `swift_package_test` via XcodeBuildMCP). Record the raw numbers: nil-reporter median (ms), Noop median (ms), delta percentage, sample size, iteration count, hardware (chip + RAM). Write these into the PR body.
5. Verify `make lint` is clean across the whole tree.
6. Verify §10 implementation checklist in `REQUIREMENTS-instrumentation.md` is fully ticked (manual checklist in the PR description; cross-reference each box to its sortie).
7. Open PR `instrumentation/01` → default branch with the recorded baseline-overhead numbers, line-number drift findings (Q3), and the Q1–Q5 decision log in the description.
8. **Do NOT cut the release tag in this sortie.** Tagging happens after merge via the release workflow with our next minor release version — drafting the tag is supervisor work post-merge.

**Exit criteria:**
- [ ] `make build` succeeds.
- [ ] `make test` passes, **including** the overhead test (subject to task 3's environment-variable gate decision recorded in the PR).
- [ ] `make lint` clean.
- [ ] PR description contains a numbered table with: `nil_median_ms`, `noop_median_ms`, `delta_percent`, `iterations`, `steps_per_iter` (= 50), `hardware`. `delta_percent` ≤ 1.0.
- [ ] PR description contains a Q1–Q5 resolution log.
- [ ] PR description contains the line-number drift report from Sorties 3, 4, 5.
- [ ] PR is open against the default branch, branch name `instrumentation/01`.
- [ ] No new files exist outside `Sources/Tuberia/Telemetry/`, `Sources/Tuberia/Pipeline/DiffusionPipeline+Telemetry.swift`, and `Tests/TuberiaTests/`.
- [ ] Every box in §10 of `REQUIREMENTS-instrumentation.md` is ticked in the PR description's checklist.

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 1 |
| Total sorties | 7 |
| Dependency structure | sequential (single PR, single branch) |
| Critical-path sortie | Sortie 1 (publishes `TuberiaTensorStat` for downstream libs) |
| Highest-risk sortie | Sortie 7 (the +1% overhead bar over 50-step × 30 iterations) |
| Release outcome | Tag next minor release version after merge |
| Open questions resolved in-plan | Q1, Q2, Q3, Q4, Q5 (see Open Questions & Missing Documentation) |
| Open questions still blocking | none |
| Maximum parallelism within plan | 1 (all sorties are supervising-agent only) |
| Cross-repo unblock event | Sortie 1 merge + minor-version tag → unblocks flux-2-swift-mlx, pixart-swift-mlx, SwiftVinetas |

## Refinement Pass Results

| Pass | Status | Changes |
|------|--------|---------|
| 1. Atomicity & Testability | PASS | 1 sortie split (old Sortie 6 → new Sortie 6 + Sortie 7), 0 merged, 12 vague exit criteria replaced with grep/file-count/build checks. |
| 2. Prioritization | PASS | Priority scores added to all 7 sorties. No reordering required — natural dependency order already aligned with priority order. |
| 3. Parallelism | PASS | Single linear critical path — no intra-WU parallelism is safe. Inter-repo parallelism (flux/pixart/SwiftVinetas after Sortie 1 merge + tag) documented for parent plan. All 7 sorties marked supervising-agent-only (every sortie has build steps). |
| 4. Open Questions & Vague Criteria | PASS | 5 questions surfaced (Q1–Q5), all resolved in-plan with recommendations recorded. 0 still blocking. |

**VERDICT**: Plan is ready to execute.

**Next step**: `/mission-supervisor start /Users/stovak/Projects/SwiftTuberia/EXECUTION_PLAN.md`
