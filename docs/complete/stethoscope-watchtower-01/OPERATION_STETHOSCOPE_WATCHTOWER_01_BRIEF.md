# Iteration 01 Brief — OPERATION STETHOSCOPE WATCHTOWER

**Mission:** Add an opt-in telemetry surface to SwiftTuberia so consumers can observe `loadModels`, `generate`, `unloadModels`, weight-loader I/O, LoRA merge events, and Metal-allocated-bytes deltas — without Tuberia depending on any specific telemetry framework.
**Branch:** `mission/stethoscope-watchtower/01`
**Starting Point Commit:** `37aeb415c9236983ad09296f8adced44be9df4a2` (chore: archive completed VANISHING MANIFEST iter-02 plan)
**Sorties Planned:** 13
**Sorties Completed:** 13
**Sorties Failed/Blocked:** 0
**Duration:** 14 commits on the mission branch (1 bootstrap + 13 sorties + 1 finalize). Every sortie completed on first attempt — zero BACKOFF, zero FATAL.
**Outcome:** Complete
**Verdict:** **Keep the code.** Ship as the next minor release version. No rollback. Architectural decisions hold; the cross-actor forwarding gap is genuinely documentation-only.

---

## Section 1: Hard Discoveries

### 1. Swift 6 forbids retroactive Codable conformance on enums-with-associated-values from outside the declaring file

**What happened:** Sortie 2 planned to add a *test-local* `Codable` extension on `TuberiaTelemetryEvent` for round-trip testing (the requirements explicitly said "production code does NOT ship `Codable`"). Swift 6 rejected this — retroactive conformance on enums with associated values must live in the declaring module/file.

**What was built to handle it:** Tests use a JSONSerialization-based dictionary round-trip via free functions in the test file. Stricter than the plan: there is no `Codable` conformance anywhere — production or test.

**Should we have known this?** Yes. Swift 6 strict-mode rules around retroactive conformance on associated-value enums are documented and have shown up before (mlx-audio-swift hit a near-identical wall). The breakdown should have inspected an existing peer's mock-event test for the round-trip strategy before specifying Codable.

**Carry forward:** When telemetry events use associated values, do not specify `Codable` round-trip as the test strategy. Specify dictionary round-trip via `JSONSerialization` from the start.

### 2. `MemoryManager.clearGPUCache` had to become `async` to emit telemetry

**What happened:** Sortie 9 needed `clearGPUCache` to call `await telemetry.capture(.unloadSegment(role: nil, ...))`. The function was previously synchronous. Making it `async` is technically a source-breaking change for any external caller.

**What was built to handle it:** All known callers are inside the actor surface and already use `await` at the boundary, so the change compiled clean and 57/57 tests passed without modifying any caller. Verified by build success and the full test suite.

**Should we have known this?** Yes — partially. The plan correctly identified the emit site but did not call out that the existing function was sync. A two-line audit (`grep -B2 "clearGPUCache" Sources/`) would have surfaced it. Low cost in retrospect, but it's the kind of thing a refinement pass should have flagged as a potential breaking-change risk.

**Carry forward:** When wiring a telemetry emit into an existing function, the breakdown must verify whether the function is sync or async, and whether async-ifying it is source-breaking for external callers. Add this as a checklist item to refine-atomicity for instrumentation sorties.

### 3. Four `throw` sites in `validateAssembly` are intrinsically un-instrumentable

**What happened:** Sortie 6 wired `errorThrown` emits before every reachable `throw PipelineError` in `DiffusionPipeline.swift`. Four throws inside `private static func validateAssembly` (called from `init`) cannot be instrumented: the function is static-sync, cannot `await emit(...)`, and runs before any consumer can call `setTelemetry(_:)` to attach a reporter.

**What was built to handle it:** Sortie 6 instrumented the 14 reachable throws (one per public-API throw site) and explicitly documented the 4 unreachable validateAssembly throws as accepted deviation. No reporter attached at construction time means no observability for assembly-validation errors anyway — the gap is honest, not hidden.

**Should we have known this?** Partially. The plan listed throw sites by line number but did not classify them by reachability. A static-sync function called from `init` is in principle an architectural smell, but refactoring it out is a separate mission, not part of this one.

**Carry forward:** For instrumentation missions, the breakdown should classify throw sites into "reachable from public API after `setTelemetry`" vs "pre-init / static-sync / unreachable" and not promise to instrument the latter. The 14-of-18 number should be the goal stated up front, not a deviation.

### 4. Cross-actor seam wiring is not implicit — `MemoryManager.shared` needs its own `setTelemetry` call

**What happened:** Sortie 9 added `setTelemetry` to `MemoryManager.shared` (a global actor singleton) so it could emit the post-cache-clear `unloadSegment(role: nil, ...)` event with Metal-bytes deltas. `DiffusionPipeline.setTelemetry` does **not** forward to `MemoryManager.shared.setTelemetry`. A consumer following the original single-call snippet would silently miss the canonical "did unload free GPU pages?" probe.

**What was built to handle it:** Sortie 13 resolved via Option A (documentation): README.md and AGENTS.md now show the two-call consumer pattern (`await pipeline.setTelemetry(...)` + `await MemoryManager.shared.setTelemetry(...)`). Auto-forwarding was rejected because (a) `MemoryManager.shared` is a singleton, so any pipeline auto-forwarding would last-writer-wins clobber other pipelines' telemetry, and (b) no other seam setter (`setComponentReadinessService`, `setMemoryGate`) cross-forwards either; auto-forwarding `setTelemetry` would be the sole exception.

**Should we have known this?** Yes. Both sides of the seam (`DiffusionPipeline` and `MemoryManager.shared`) are actor types that hold their own state independently. This is exactly the kind of thing the requirements doc should have specified: either "single setter forwards to both" or "consumer attaches twice". Instead it specified the two emit sites and left the wiring ambiguous, which forced the decision into Sortie 13 under acceptance pressure.

**Carry forward:** When telemetry spans multiple actor types, the requirements doc must explicitly state the consumer wiring contract. Multi-actor seams need a specified wiring policy at breakdown time, not at acceptance time.

---

## Section 2: Process Discoveries

### What the Agents Did Right

#### 1. In-target stub-recipe pattern held across four test sorties

**What happened:** The plan specified a `private struct ...StubRecipe: PipelineRecipe, Sendable` inside each test file (mirroring `RecipeRoleMapTests.swift` and `MemoryGuardTests.swift`) instead of a shared `MockPipelineRecipe` (which lives in `TuberiaGPUTests` and is invisible to `TuberiaTests`). All four test-writing sorties (4, 7, 8, 10) used it cleanly.

**Right or wrong?** Right. The convention prevented a cross-target visibility tangle and kept each test file self-contained. Zero cross-test coupling.

**Evidence:** 4 test files (`DiffusionPipelineTelemetryTests.swift`, `GenerateTelemetryTests.swift`, `LoRAErrorTelemetryTests.swift`, `WeightLoaderTelemetryTests.swift` / `MemoryManagerTelemetryTests.swift`) each declare their own private stub. Total: 528 + 367 + 514 + 162 + 190 = 1761 LOC of test code, no shared scaffolding, all 81 tests pass.

**Carry forward:** When a test target cannot see a richer mock from a sibling test target, prefer a per-file inline private stub over fighting the visibility — and lock the convention in the plan so every downstream test sortie does it the same way.

#### 2. Sergeant principle (one objective per sortie) held throughout

**What happened:** Every sortie had exactly one deliverable. Sortie 5 was "wire generate happy path + denoiseStep" (one cohesive code seam). Sortie 6 was "wire LoRA + errorThrown" (one cohesive code seam). The plan deliberately split the original 11-sortie design into 13 (5→5+6, 6→7+8) when refinement showed dual goals.

**Right or wrong?** Right. Zero retries, zero context overruns reported, every sortie's exit criteria were specific and machine-verifiable.

**Evidence:** 13/13 sorties COMPLETED on first attempt. Zero BACKOFF entries in the Decisions Log. Zero FATAL escalations.

**Carry forward:** Refinement pass 1 (atomicity) earned its keep here. The 11→13 split looked like over-planning at the time but was correct — keep doing it.

### What the Agents Did Wrong

#### 1. Sortie 1 imported `Foundation` despite no Foundation symbols being used

**What happened:** Sortie 1 included `import Foundation` in the new telemetry files because the spec showed it. The current event-enum cases use no Foundation symbols — they're all primitives and `String` (which is in the standard library, not Foundation).

**Right or wrong?** Mildly wrong. Cargo-culted from the spec. Non-blocking, no behavioral effect, but a working agent could have noticed.

**Evidence:** `Sources/Tuberia/Telemetry/TuberiaTelemetryEvent.swift` opens with `import Foundation` while using only `String`, `Int`, `Double`, `Bool`, `Sendable`, and `Error`. The agent flagged it as "non-blocking" — correctly — but should have removed the import.

**Carry forward:** Agent prompts for foundation-type sorties should include "Only import what you use; do not cargo-cult `import Foundation` from the spec."

#### 2. Sortie 3 introduced a file-scope helper class to satisfy a Swift 6 constraint

**What happened:** Sortie 3 needed a mutable box to track download-progress rate-limiting inside a generic function. Swift 6 forbids generic-function-body-nested classes, so the agent created `_DownloadProgressState` at file scope.

**Right or wrong?** Acceptable but pattern-leaky. A file-scope helper for a one-off rate-limit counter is heavier than needed; an `actor`-isolated stored property or an `@unchecked Sendable` struct closer to the call site would have been less surface-area. Non-blocking — no behavioral issue and tests pass.

**Evidence:** Identifier `_DownloadProgressState` declared at file scope in `DiffusionPipeline.swift` (visible in commit `3975f08`). Used only by the rate-limited download-progress emit logic.

**Carry forward:** When a Swift 6 constraint forces a structural change, the agent should choose the *narrowest* refactor (closure-captured `var` in a `let counter = ...` outside; or actor-isolated state). Adding a file-scope type is the broadest move.

### What the Planner Did Wrong

#### 1. Plan grep for Sortie 5 was over-precise (renderStart pattern)

**What happened:** Sortie 5's exit criterion grep for `\.renderStart\(` failed because `renderStart` is a no-arg enum case — the correct Swift is `emit(.renderStart)`, not `emit(.renderStart())`. The plan-author wrote a grep that assumed every event case takes payload.

**Right or wrong?** Planner error. The grep was wrong; the code was right. Sortie 5's agent correctly diagnosed and surfaced the plan-grep bug rather than blindly editing code to satisfy it.

**Evidence:** SUPERVISOR_STATE.md Sortie 5 COMPLETED entry: "Plan exit-criterion grep `\.renderStart\(` is over-precise (renderStart is a no-arg case → `emit(.renderStart)` is correct Swift) — accepted as legitimate plan-grep bug, not missing emit."

**Carry forward:** Refinement-pass 1 (atomicity) should also include a "verify exit-criteria greps are correct" step. Run the grep against the *current* codebase before locking the plan; if the pattern doesn't match expected sites, the grep is wrong.

#### 2. Sortie 12 (Documentation) had an undocumented soft dependency on Sortie 2

**What happened:** Sortie 12's entry criteria listed only Sorties 1 and 3, but task 4 required referencing `Tests/TuberiaTests/Telemetry/MockTuberiaTelemetryReporter.swift` — a file created in Sortie 2. The supervisor caught this at dispatch time and held Sortie 12 until Sortie 2 finished.

**Right or wrong?** Planner missed it; supervisor caught it. Mild planning error, well-recovered. Cost: Sortie 12 started slightly later than the dependency graph allowed.

**Evidence:** SUPERVISOR_STATE.md Decisions Log: "Sortie 12 entry criteria list only Sorties 1+3, but task 4 references `Tests/TuberiaTests/Telemetry/MockTuberiaTelemetryReporter.swift` (created in Sortie 2). Soft dependency on 2 — dispatching 12 after 2 completes so README references a real file."

**Carry forward:** When a sortie's *task body* references a file/symbol, the planner must add the producing sortie to the dependency list — not just the conceptual dependency. Refinement pass 3 (parallelism) should grep each sortie body for file paths and verify each references a file produced by an upstream sortie.

#### 3. Cross-actor wiring decision pushed to Sortie 13 instead of being decided at breakdown

**What happened:** The architectural question "does pipeline.setTelemetry forward to MemoryManager.shared.setTelemetry?" was not answered until Sortie 13 (the acceptance sortie). It should have been answered in §13 of REQUIREMENTS-telemetry.md alongside the other four locked decisions.

**Right or wrong?** Planning gap. Sortie 13 made the right call (Option A: documentation), but the decision under acceptance pressure is the worst time to make architectural choices.

**Evidence:** Sortie 13 commit `a72771f` includes a 60-line "Known Gap — Cross-Actor Telemetry Forwarding" section in MISSION_ACCEPTANCE.md. That entire analysis should have been front-loaded.

**Carry forward:** When telemetry seams span multiple actor types, the requirements/breakdown phase must specify the consumer wiring contract explicitly. Add a "multi-actor seam wiring" checklist item to the breakdown command for instrumentation missions.

---

## Section 3: Open Decisions

### 1. Should `validateAssembly` be refactored out of `init` so its 4 throws can emit `errorThrown`?

**Why it matters:** Currently 4 of 18 throw sites in `DiffusionPipeline.swift` cannot emit telemetry. A consumer monitoring construction-time validation failures has no observability. For most users this is fine (validation errors are surfaced via the throw), but for diagnostic tooling (e.g., a CI harness running many assemblies in a loop) the gap is real.

**Options:**
- **A. Leave it.** Document that pre-`setTelemetry` errors are not observable. Cost: 0. Consumer impact: zero observability for a small set of throws that are mostly programmer errors anyway.
- **B. Refactor `validateAssembly` to be called after init via `await pipeline.validate()`.** Cost: source-breaking change to the public init contract. Consumer impact: every existing consumer must add a `validate()` call. Adds an `unvalidated` window.
- **C. Provide a static factory (`DiffusionPipeline.assembled(from:)`) that takes an optional reporter at construction time.** Cost: new public API. Consumer impact: opt-in; existing consumers untouched.

**Recommendation:** **A.** The four un-instrumentable throws are programmer errors (mismatched component IDs, missing required components). They surface clearly via `throw PipelineError`, which any caller already handles. The diagnostic value of telemetry on these specific throws is low relative to the API churn of B or C. Revisit if a downstream consumer asks for it.

### 2. Should `TuberiaTelemetryEvent` carry a `pipelineId: UUID`?

**Why it matters:** Decisions Locked from Source §13 deferred this. With multiple concurrent pipelines (test suites, future host apps running parallel generations), a reporter cannot distinguish events from different pipelines without a pipeline ID. Today this is an in-process single-pipeline assumption; multi-pipeline is not blocked, but its observability is.

**Options:**
- **A. Defer.** Continue to assume one pipeline per process for telemetry purposes.
- **B. Add `pipelineId: UUID` to every event in the next minor bump (post-this-mission).** Additive; zero source-breakage if `init` generates the UUID and the field is non-optional.
- **C. Add `pipelineId: UUID?` (optional) to every event now, defaulted to `nil`.** Smaller bump; ergonomically worse (every consumer has to handle the optional).

**Recommendation:** **A** for this mission, **B** for the next. The current consumers (Produciesta, pixart-swift-mlx, flux-2-swift-mlx) all run a single pipeline. Adding `pipelineId` now is speculative. When a third Produciesta-class consumer surfaces with a multi-pipeline use case, do B as a coordinated minor-bump alongside the consumer.

---

## Section 4: Sortie Accuracy

| Sortie | Task | Model | Attempts | Accurate? | Notes |
|--------|------|-------|----------|-----------|-------|
| 1 | Foundation types | sonnet | 1 | Yes | Output survived to final state untouched. Minor `import Foundation` cargo-cult flagged; non-blocking. |
| 2 | Test infra (mock + event tests) | sonnet | 1 | Yes | Codable→JSONSerialization swap was a discovery, not rework. Output stable. |
| 3 | DiffusionPipeline load/unload instrumentation | sonnet | 1 | Yes | 15 emit sites, all survived. `_DownloadProgressState` file-scope helper is the only structural deviation. |
| 4 | DiffusionPipeline lifecycle tests | sonnet | 1 | Yes | 8 `@Test` functions, no rework. In-target stub pattern landed cleanly. |
| 5 | Generate happy path + denoiseStep | sonnet | 1 | Yes | 9 emit sites at the expected lines. Plan-grep bug surfaced (planner fault, not sortie fault). |
| 6 | LoRA + errorThrown instrumentation | sonnet | 1 | Yes | 14 errorThrown + 3 LoRA emits. 4 validateAssembly throws correctly identified as un-instrumentable. |
| 7 | Generate / denoiseStep tests | sonnet | 1 | Yes | 3 `@Test` functions, no rework. |
| 8 | LoRA / errorThrown tests | sonnet | 1 | Yes | 5 `@Test` functions, no rework. |
| 9 | WeightLoader + MemoryManager instrumentation | **opus** | 1 | Yes | Highest-complexity sortie (Metal API surface). Opus selection paid off — first-attempt success. Cross-actor wiring gap surfaced here (planning fault, not sortie fault). |
| 10 | WeightLoader + MemoryManager tests | sonnet | 1 | Yes | 8 tests / 2 suites, no rework. |
| 11 | Performance regression suite + Makefile | sonnet | 1 | Yes | Delta -88.94% (well under 10% threshold). Compile-flag gating worked. |
| 12 | Documentation | **haiku** | 1 | Yes | Markdown-only sortie. Haiku was correct cost choice; output stable. |
| 13 | Acceptance verification | sonnet | 1 | Yes | 81 tests pass. Made the cross-actor wiring decision (Option A). MISSION_ACCEPTANCE.md is comprehensive. |

**Accuracy summary:** 13/13 sorties accurate. Zero output was overwritten or invalidated by a downstream sortie. Zero retries. The model-selection table held: haiku for markdown, sonnet for mechanical instrumentation/tests, opus only for the Metal-API/defaulted-param risk surface.

---

## Section 5: Harvest Summary

What we now know that we didn't know before: the SwiftSecuencia/mlx-audio-swift telemetry pattern (protocol + `setTelemetry` + `private emit` helper + nil reporter ⇒ zero events) ports cleanly to a multi-actor surface, but **the consumer wiring contract must be specified at breakdown time, not at acceptance time**. The single most important thing that changes about the next iteration: when a telemetry surface spans more than one actor, the requirements doc must lock the wiring (forward / two-call / factory) before sorties dispatch. Sortie 13 had to make that call under acceptance pressure; it landed correctly, but only because the documentation-only fix was cheap. A future mission with a less-cheap fix (e.g., one that required a second public type or a source-breaking init change) would have wedged.

The mission is also a clean datapoint that **opus-on-the-critical-risk-sortie + sonnet-everywhere-else + haiku-for-markdown** is an effective cost mix for a 13-sortie instrumentation mission. 12 sonnet + 1 opus + 1 haiku, all first-attempt success.

---

## Section 6: Files

### Preserve (read-only reference for next iteration)

| File | Branch | Why |
|------|--------|-----|
| `OPERATION_STETHOSCOPE_WATCHTOWER_01_BRIEF.md` | `mission/stethoscope-watchtower/01` (post-clean: `docs/complete/stethoscope-watchtower-01/`) | This brief. Authoritative post-mission record. |
| `MISSION_ACCEPTANCE.md` | `mission/stethoscope-watchtower/01` (post-clean: archived) | §12 acceptance ticks + cross-actor wiring decision rationale. |
| `EXECUTION_PLAN.md` | `mission/stethoscope-watchtower/01` (post-clean: archived) | The plan that worked. Useful as a template for future instrumentation missions. |
| `SUPERVISOR_STATE.md` | `mission/stethoscope-watchtower/01` (post-clean: archived) | Per-sortie Decisions Log; first-attempt-success pattern is the reference data for future model-selection. |
| `REQUIREMENTS-telemetry.md` | `mission/stethoscope-watchtower/01` | Source of truth. Carry forward; revisit §13 to add multi-actor wiring policy before any future telemetry mission. |

### Discard (will not exist after rollback)

**Verdict is "keep the code" — no rollback.** Nothing to discard. The mission branch will be merged or kept as the integration branch for the next minor release.

| File | Why it's safe to lose |
|------|----------------------|
| (none) | Mission verdict is COMPLETE — keep all work. |

---

## Iteration Metadata

**Starting point commit:** `37aeb415c9236983ad09296f8adced44be9df4a2` (`chore: archive completed VANISHING MANIFEST iter-02 plan`)
**Mission branch:** `mission/stethoscope-watchtower/01`
**Final commit on mission branch:** `786a684cd8a5d24d51fe3076d23ad86bcba67118` (`chore(mission): finalize SUPERVISOR_STATE.md — mission complete`)
**Rollback target:** N/A — verdict is "keep the code"
**Next iteration branch:** N/A — no next iteration planned for this operation. The two open decisions (validateAssembly refactor, pipelineId UUID) are tracked here for whenever they become real.
