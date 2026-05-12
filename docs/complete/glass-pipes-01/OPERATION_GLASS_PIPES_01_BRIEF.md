---
name: OPERATION_GLASS_PIPES_01_BRIEF
mission: swift-tuberia-instrumentation
feature_name: OPERATION GLASS PIPES
iteration: 1
state: completed
---

# Iteration 01 Brief — OPERATION GLASS PIPES

**Mission:** Instrument SwiftTuberia's five protocol seams with a zero-cost-when-nil telemetry surface and publish the cross-repo `TuberiaTensorStat` Codable type that downstream MLX libraries depend on.
**Branch:** `instrumentation/01`
**Starting Point Commit:** `b6f3ba6` (`docs: fix internal links after archive move`)
**Sorties Planned:** 7 (+1 mid-flight patch sortie inserted as 3.5)
**Sorties Completed:** 8 (1, 2, 3, 3.5, 4, 5, 6, 7)
**Sorties Failed/Blocked:** 0 (Sortie 7's overhead test was honestly invalidated and removed — not a sortie failure)
**Duration:** 1 calendar day (2026-05-12), 9 sortie commits + 7 supervisor checkpoint commits, all opus/sonnet (no haiku in plan).
**Outcome:** Complete
**Verdict:** **KEEP THE CODE.** All five protocol seams are instrumented per `REQUIREMENTS-instrumentation.md` §3–§5, 47 tests pass locally, and the structural zero-cost-when-nil discipline is verifiable by grep. Roll forward into release.

---

## Section 1: Hard Discoveries

### 1. `validateAssembly` runs synchronously inside `init`, but `setTelemetry(_:)` is post-init

**What happened:** Sortie 3 wired `pipelineConfigured` + six `assemblyCheckPassed` events inside `DiffusionPipeline.init(recipe:)`. The agent surfaced honestly that these events were structurally unobservable: `setTelemetry(_:)` had to arrive *after* `init` returned, by which point the events had already fired into `nil`. The Sortie 6 "all six assembly checks fire" test was impossible to write against the original API.
**What was built to handle it:** Sortie 3.5 (a 3-line patch sortie inserted by the supervisor mid-flight) added a defaulted `telemetry:` parameter to `init(recipe:telemetry:)`. Existing `init(recipe:)` callers compile unchanged.
**Should we have known this?** Yes. A 10-minute trace from "events emitted in init" to "telemetry settable post-init" would have caught it during planning. Pass 1 (atomicity) didn't catch it because it looked at *file boundaries*, not at *temporal observability*.
**Carry forward:** When a protocol emits events from a constructor, the telemetry sink must be injectable at construction time. Don't ship "set the sink later" APIs for objects that fire events during init.

### 2. Swift cooperative executor does not guarantee inter-Task ordering for `Task { }` from a synchronous context

**What happened:** Each of the six `validateAssembly` checks emits its event via `Task { await reporter.capture(...) }` because `validateAssembly` is synchronous and `init` cannot be `async` without breaking the public ABI. Sortie 6 row 1 originally required "all six events fire in order" — Sortie 6 agent surfaced that this is structurally unenforceable: Swift's executor schedules detached Tasks without total ordering across them.
**What was built to handle it:** REQUIREMENTS §7 row 1 was updated (with user concurrence) to require *presence + count == 6*, not order. Tests identify each event by its `AssemblyCheck` kind, not its position. **Production code was NOT patched** to fake ordering — the spec was the wrong one.
**Should we have known this?** Yes. Anyone who has written `Task { }` from sync code knows the order is "happens-before only inside one task chain." We didn't write down the constraint, the planner asked for an impossible test, the agent caught it.
**Carry forward:** Any sortie that asserts "events fire in order" must first establish whether those events are emitted from the same actor context. If detached Tasks are involved, drop ordering claims at the planning stage.

### 3. `if let telemetry { ... }` guards make Noop-vs-nil overhead unmeasurable

**What happened:** §5 of the requirements chose explicit `if let telemetry { ... }` guards over an `@autoclosure`-based protocol. Under that design, a `NoopTuberiaTelemetryReporter` still runs every guarded `TuberiaTensorStat.sample()` (8 MLX reductions per call) and every `await reporter.capture(...)` (async hop). Sortie 7 measured Noop vs. nil honestly — **+1381% delta** (nil median 11.59 ms, Noop median 171.76 ms on Apple M2 Pro / 32 GB). The agent stopped before opening the PR.
**What was built to handle it:** Test removed entirely. REQUIREMENTS §7 row 5 deleted. Zero-cost-when-nil is now proven *structurally*: 21 `TuberiaTensorStat.sample()` sites, every one inside an `if let telemetry { ... }` block (grep-verifiable), plus 5 functional tests covering every hot-path emission code path.
**Should we have known this?** Yes — but not at plan time, because the design choice (guards vs. autoclosure) was a Sortie-5 implementation decision. The mistake was carrying the §7 row 5 test through *after* §5 settled on guards. The two are incompatible: guards measure "cost when ON," not "cost when OFF."
**Carry forward:** When the design changes the meaning of a test, delete the test. Do not downgrade to "informational" or env-gate to dodge CI — that is exactly what was avoided here, and is on file in [[feedback_no_flaky_or_informational_tests]].

### 4. Anchor line numbers in the requirements drifted by hundreds of lines

**What happened:** REQUIREMENTS-instrumentation.md §5 cited specific lines in `DiffusionPipeline.swift` (`:480`, `:511`, `:545`, `:552`, `:560`, …). By Sortie 5 those anchors had drifted +520..+810 lines because every preceding sortie added emission code to the same file. Q3 was raised in plan refinement and the resolution was "each sortie greps for the anchor pattern."
**What was built to handle it:** Each sortie verified anchors by grepping for known patterns (e.g. `.asType(.float32)` for the CFG cast, `scheduler.configure` for scheduler-config) and recorded the drift in the PR description. The PR body contains a consolidated drift table.
**Should we have known this?** Yes. Any plan that cites line numbers in a file the plan itself will modify is a plan that will rot. The Q3 resolution was correct but should have been "stop citing line numbers, cite grep patterns" from the start.
**Carry forward:** Never put line numbers in a plan that mutates the file. Use grep anchors only.

### 5. Swift 6 strict concurrency blocks plain mutable static `var`

**What happened:** Q2 required a runtime-overridable default threshold (`TuberiaTensorStat.defaultOutOfRangeThreshold = 1e6`). The natural form `public static var defaultOutOfRangeThreshold: Double = 1e6` is forbidden under Swift 6 strict concurrency. `let` would have made the spec's "override at runtime" intent dead. Wrapping in an actor would have forced every call site into async or risked deadlock during sampling.
**What was built to handle it:** `public nonisolated(unsafe) static var defaultOutOfRangeThreshold: Double = 1e6` — accepted as a pragmatic violation, documented in PR body, kept narrow to one threshold value. Recorded in Decisions Log.
**Should we have known this?** Yes — any Swift 6 codebase has this problem. The planner didn't think about how Q2 would land in the type system.
**Carry forward:** When a plan asks for a "configurable default," call out the concurrency model up front. `nonisolated(unsafe)` is acceptable for narrow scalar defaults; actor isolation is acceptable for anything that mutates collections.

---

## Section 2: Process Discoveries

### What the Agents Did Right

#### 1. Honest reporting on Sortie 7's overhead test

**What happened:** Sortie 7 measured +1381% delta and STOPPED before opening the PR. Did not relax the assertion, did not env-gate, did not downgrade to "informational." Surfaced the structural impossibility to the supervisor.
**Right or wrong?** Exactly right. The alternative — ship a passing-by-accident test or a CI-skipped test — would have left a landmine in the codebase. Confirmed by [[feedback_no_flaky_or_informational_tests]].
**Evidence:** Commit `33bef96` body. Supervisor accepted the report; commit `62b86ea` removed the test and updated REQUIREMENTS.
**Carry forward:** This pattern is the standard. Brief authors of future sorties on it explicitly.

#### 2. Anchor-grep verification on every drifty sortie

**What happened:** Sorties 3, 4, 5 each opened with a `grep` for their anchor patterns and recorded line numbers in the commit body. All three documented significant drift.
**Right or wrong?** Right. Saved at least three rework cycles.
**Evidence:** PR body's "Q3 — Line-number drift" table.
**Carry forward:** Make it a default sortie-opener for any plan that modifies a file the plan also cites.

#### 3. Sortie 3 surfaced the init-observability gap as a question, not a fix

**What happened:** Sortie 3 found that `pipelineConfigured` and assembly events were unobservable. Agent did NOT silently patch around it (e.g. by buffering events). Reported to supervisor and waited.
**Right or wrong?** Right. Supervisor + user picked the smallest fix (defaulted init param) and dispatched a 3-line patch sortie. Buffering would have been wrong work.
**Evidence:** Decisions Log entry "Sortie 3 COMPLETED — architectural gap surfaced."
**Carry forward:** Sortie agents should report design-level surprises to the supervisor instead of inventing fixes. The supervisor knows the cross-repo blast radius; the sortie agent doesn't.

### What the Agents Did Wrong

#### 1. Sortie 6 originally asserted strict event ordering

**What happened:** Sortie 6 agent took §7 row 1 ("all six events fire in order") at face value and tried to write the test. Discovered mid-implementation that Task-dispatch from sync init makes this unenforceable. Then asked the supervisor for guidance.
**Right or wrong?** The catch was right; the late catch was wrong. A 30-second think about "where are these events emitted from" before writing the test would have caught it sooner.
**Evidence:** Mid-flight decision B in PR body.
**Carry forward:** Add a "where does this event come from" pre-flight check to test sorties when the spec asserts ordering.

#### 2. Sortie 5's DType helper duplication

**What happened:** Sortie 5 added `tuberiaPipelineCanonicalDTypeString` at `DiffusionPipeline.swift:16`, a 15-case dtype switch that duplicates the private `TuberiaTensorStat.canonicalDTypeString(_:)` from Sortie 1.
**Right or wrong?** Wrong. The agent should have lifted the existing helper to `internal` and reused it. This is on the post-merge cleanup queue but should not have shipped.
**Evidence:** SUPERVISOR_STATE.md Post-merge cleanup queue, item 1.
**Carry forward:** Sortie briefings for files that depend on earlier sorties should include an explicit "check what's already private/internal that you can lift" reminder.

### What the Planner Did Wrong

#### 1. Pass 1 (atomicity) didn't catch the init/setTelemetry race

**What happened:** The plan structured Sortie 1 (publish types), Sortie 2 (inject setters + ivars), Sortie 3 (emit events including `pipelineConfigured` in init). Pass 1 evaluated each sortie's testability *in isolation*. It did not ask "can Sortie 3's emissions be observed by Sortie 6's tests, given Sortie 2's API?"
**Right or wrong?** Wrong, predictably. Cost was the 3.5 patch sortie (~10 minutes of supervisor + agent time). Not catastrophic, but trivially avoidable.
**Evidence:** Sortie 3.5 commit `6ec1177`.
**Carry forward:** Pass 1 atomicity check should include a temporal-observability sub-check: "for every event this sortie emits, is the sink installable before the event fires, given prior sorties' API?"

#### 2. §7 row 5 (Noop-overhead test) was never compatible with §5 (guards)

**What happened:** Requirements §5 chose guard discipline (`if let telemetry`) over autoclosure-on-protocol. Requirements §7 row 5 asked for Noop-vs-nil overhead ≤ ±1%. These are incompatible: guards make Noop do real work; only `@autoclosure` (which §5 rejected) makes Noop-vs-nil approach zero. Planner kept both requirements through all 4 refinement passes.
**Right or wrong?** Wrong. Pass 2 (priority) was the right place to catch it: this test had risk 3 (highest in the plan) but was measuring the wrong thing.
**Evidence:** Sortie 7 +1381% delta; subsequent test removal.
**Carry forward:** When two requirement sections express different design choices, refinement Pass 4 (questions) must surface the conflict. If §5 picks guards, §7's overhead test must be reformulated to measure "telemetry-off baseline vs. un-instrumented baseline" — not "Noop vs. nil."

#### 3. Plan cited line numbers in a file the plan modifies

**What happened:** §5's emission table cited `:480`, `:511`, etc. Drift was inevitable.
**Right or wrong?** Wrong. Q3 in the plan tried to patch this with "each sortie greps for anchors" — which worked, but means the line numbers in the plan are dead weight after Sortie 1.
**Evidence:** Drift table in PR body (anchors moved +117..+810).
**Carry forward:** Plans should cite grep patterns, not line numbers, for files the plan will mutate. Line numbers are acceptable only for read-only references (e.g. another repo's file).

#### 4. Plan didn't think about Swift 6 strict concurrency for Q2

**What happened:** Q2 asked for a configurable threshold; planner did not consider that `static var` is forbidden under Swift 6. Sortie 1 agent had to make the call live.
**Right or wrong?** Marginal — `nonisolated(unsafe)` was an acceptable accept, but the call could have been made by the planner instead of the executor.
**Carry forward:** When a plan asks for a "configurable default," the planner specifies the concurrency container (var / actor / nonisolated(unsafe) / async getter) — don't push that call to the executor.

---

## Section 3: Open Decisions

### 1. Post-merge cleanup — DType helper duplication

**Why it matters:** Two implementations of the same 15-case dtype switch will drift. Low-severity but inevitable.
**Options:** A) Lift `TuberiaTensorStat.canonicalDTypeString` to `internal` and delete the duplicate. B) Leave both; document. C) Accept duplication forever.
**Recommendation:** A. One sortie's worth of work, done post-merge or in the next iteration. Already on the cleanup queue.

### 2. Post-merge cleanup — `FlowMatchEulerScheduler.predictionType`

**Why it matters:** Inherits `"unknown"` from the protocol default. Flow-matching has real prediction semantics (`flow_match` / `velocity`); the default string is wrong information rather than no information.
**Options:** A) Add `"flow_match"` override. B) Leave `"unknown"`; document that flow-matching consumers should ignore the field. C) Remove the field for flow-matching schedulers (would require protocol restructuring).
**Recommendation:** A. Trivial fix, real value to telemetry consumers. Could ship in this PR if reviewer asks; otherwise next iteration.

### 3. The +1% structural-proof claim needs an artifact

**Why it matters:** The PR says "zero-cost-when-nil is proven structurally." That claim currently lives in the PR description as English. There is no scripted check that fails CI if a new `TuberiaTensorStat.sample(` call lands outside an `if let telemetry` block.
**Options:** A) Add a lint script (`scripts/check-telemetry-guards.sh`) that greps `Sources/Tuberia/Pipeline/DiffusionPipeline.swift` and asserts every `sample(` line is inside an `if let telemetry` block. B) Accept that future sorties must hand-check the discipline. C) Write a SwiftSyntax-based check (over-engineering for this).
**Recommendation:** A. Half-an-hour sortie in the next iteration of this campaign (the cross-repo Vinetas work). Cheap insurance for downstream callers that copy the pattern.

### 4. CI is currently red on `tests.yml` due to upstream `swift-tokenizers` 0.6.1 / TokenizersFFI RustBuffer compile failure

**Why it matters:** PR #35's CI failure is `Cannot find type 'RustBuffer' in scope` in `swift-tokenizers/TokenizersFFI`. Same failure exists on PR #34 (development → main) since 2026-05-12. **This is NOT a GLASS PIPES regression.** Local builds pass; the upstream Rust-FFI binding files are missing in the CI environment. Last known-good CI run was 2026-05-08.
**Options:** A) Investigate `swift-tokenizers` 0.6.1 — likely needs a Rust toolchain step in the CI workflow or a release-asset that includes the pre-built `.xcframework`. B) Pin to an older `swift-tokenizers` until upstream is fixed. C) Merge into `development` (unprotected) and let `development → main` (protected) wait for the upstream fix.
**Recommendation:** C for the immediate merge of GLASS PIPES into `development`, then A as a separate investigation. Do not bypass `main`'s branch protection.

---

## Section 4: Sortie Accuracy

| Sortie | Task | Model | Attempts | Accurate? | Notes |
|--------|------|-------|----------|-----------|-------|
| 1 | Publish `TuberiaTensorStat` + telemetry surface | opus | 1/3 | YES | Two pragmatic accepts (`nonisolated(unsafe)`, pre-reduction f32 cast); both survived into final state. Public API stable. |
| 2 | Injection points (setters + defaulted params) | opus | 1/3 | PARTIAL | `memoryGate` type widening was wider than plan asked. Accepted, but the legacy `setMemoryGate(_:)` seam now silently drops telemetry — documented inline. Not a regression but a wider API surface than necessary. |
| 3 | Lifecycle/assembly/memory/weight/LoRA emission | sonnet | 1/3 | YES — but surfaced init gap | Survived intact. The "init events are no-ops" surfacing was *not* a bug in this sortie's work — it was an accurate report that triggered Sortie 3.5. |
| 3.5 | Defaulted `telemetry:` param on init | sonnet | 1/3 | YES | 3-line patch. Smallest possible fix. Source-compatible. |
| 4 | Text-encoder + scheduler emission | sonnet | 1/3 | YES | Added `Scheduler.predictionType` protocol default — flow-matching schedulers inherit `"unknown"` (cleanup item). |
| 5 | Hot-path emission (denoise/CFG/backbone/decoder/renderer) | opus | 1/3 | YES — minor surface | 17 emission sites + 17 anomaly invocations, all properly guarded. DType helper duplication (cleanup queue) is the one wart. |
| 6 | Functional tests (5 files + recorder) | sonnet | 1/3 | YES — and surfaced two spec drifts | Refused to silently patch production for assembly-ordering and backbone-phase-string drifts. Forced REQUIREMENTS updates instead. Strong sortie. |
| 7 | Noop overhead measurement | sonnet | 1/3 | TEST REMOVED | Honest measurement, honest stop. The test was never compatible with §5's guard discipline. Removal was correct, not a sortie failure. |

**Aggregate:** 8/8 sorties produced code that survived. 0 reverts, 0 retries, 0 FATAL escalations. The one mid-flight insertion (3.5) was a planner-side gap, not a sortie failure.

---

## Section 5: Harvest Summary

The single most important thing learned: **when a plan emits events from a constructor, the telemetry sink must be injectable at construction time.** This is the one architectural mistake that bled into a patch sortie. The other lessons (Task ordering, guard vs. autoclosure, line-number drift, Swift 6 concurrency) are all *known* problems that the planner should have anticipated — they cost time because they were caught downstream, not because they were unknowable.

For the cross-repo Vinetas campaign (parent plan): the `TuberiaTensorStat` Codable contract is stable. Downstream libraries (`flux-2-swift-mlx`, `pixart-swift-mlx`, `SwiftVinetas`) can begin their slices as soon as the minor-version tag is cut. They should mirror this repo's `if let telemetry` discipline; the structural-proof check from Open Decision 3 should be the first sortie of the next iteration of this work unit.

---

## Section 6: Files

### Preserve (read-only reference for next iteration)

| File | Branch | Why |
|------|--------|-----|
| `Sources/Tuberia/Telemetry/TuberiaTensorStat.swift` | `instrumentation/01` | Cross-repo published Codable type. Reference for downstream slices. |
| `Sources/Tuberia/Telemetry/TuberiaTelemetryEvent.swift` | `instrumentation/01` | 22-case event enum + 5 nested enums. Reference for consumers. |
| `Sources/Tuberia/Telemetry/TuberiaTelemetryReporter.swift` | `instrumentation/01` | Protocol + Noop reporter. Reference for consumers' adapter implementations. |
| `Sources/Tuberia/Pipeline/DiffusionPipeline+Telemetry.swift` | `instrumentation/01` | `setTelemetry`, `emitAnomalyIfPresent`. Reference for the guard-discipline pattern. |
| `Tests/TuberiaTests/Support/RecordingTelemetryReporter.swift` | `instrumentation/01` | Test-fixture pattern. Copy to downstream test targets verbatim. |
| `REQUIREMENTS-instrumentation.md` | `instrumentation/01` | Updated §7 row 1, row 4, row 5, §10 reflect mid-flight decisions B/C/D. This is the canonical contract. |

### Discard (will not exist after rollback)

| File | Why it's safe to lose |
|------|----------------------|
| _(none — verdict is KEEP)_ | The mission is rolling forward, not rolling back. No rollback ritual. |

---

## Section 7: Iteration Metadata

**Starting point commit:** `b6f3ba6` (`docs: fix internal links after archive move`)
**Mission branch:** `instrumentation/01`
**Final commit on mission branch (pre-brief):** `1c7e01d` (`mission(GLASS PIPES): mission COMPLETED — PR #35 open`)
**Rollback target:** N/A — verdict is keep the code.
**Next iteration branch:** N/A for this repo. The next iteration of the *parent campaign* (cross-repo Vinetas) begins in `flux-2-swift-mlx`, `pixart-swift-mlx`, `SwiftVinetas` once the minor tag is cut here.

**Post-merge work owed:**
1. Merge PR #35 (`instrumentation/01` → `development`).
2. Ensure PR #34 (`development` → `main`) is ready for review with updated body reflecting GLASS PIPES contents.
3. Once #34 merges, tag `main` with our next minor release version so downstream libs can pin.
4. Address Open Decisions 1, 2, 3 as follow-up sorties (next iteration of the cross-repo campaign).
5. Investigate Open Decision 4 (upstream `swift-tokenizers` 0.6.1 / TokenizersFFI CI red) as a separate, non-GLASS-PIPES task — it is blocking BOTH PR #35 and PR #34 currently.
