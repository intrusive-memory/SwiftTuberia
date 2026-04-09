# Iteration 01 Brief — OPERATION BLUNT BLADE REFORGE

**Mission:** Purge 6 redundant tests across three Swift test targets and reforge the SwiftTuberia test suite with 26 new precision unit tests covering scheduler math, tensor transforms, pixel conversion, LoRA key conventions, pipeline error guards, and device capability contracts.
**Branch:** `mission/blunt-blade-reforge/01`
**Starting Point Commit:** `c3840ebf54379f10ff21554938ea67eb7b470dee`
**Sorties Planned:** 7
**Sorties Completed:** 7 + 1 unplanned fix (PipelineAssemblyTests)
**Sorties Failed/Blocked:** 0 (one temporary BLOCKED on TuberiaGPUTests; resolved by unplanned sortie)
**Outcome:** Complete
**Verdict:** Keep the code. All three test targets pass. One production file (`DiffusionPipeline.swift`) has a testing hook that warrants review before merge.

---

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.
> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.
> **Work Unit** — A grouping of sorties (package, component, phase).

---

## Section 1: Hard Discoveries

### 1. `transformer` Is Private — `@testable import` Does Not Help

**What happened:** Sortie 1 wrote `let embeddingWeight = encoder.transformer.embedding.weight` to verify weights were applied. `T5XXLEncoder.transformer` is `private var`, not `internal var`. `@testable import` only elevates `internal` to visible — it cannot bypass `private`. The SourceKit diagnostic fired immediately.

**What was built to handle it:** The assertion was replaced with checks on `encoder.isLoaded == true` and `encoder.currentWeights != nil`, both of which are public API. The sortie self-corrected before xcodebuild ran.

**Should we have known this?** Yes. Reading `T5XXLEncoder.swift` property visibility before writing the test would have revealed it immediately. The plan said "access internal state" without specifying which properties were internal vs. private.

**Carry forward:** Before writing any assertion that accesses component internals, read the source file and annotate which properties are `public`, `internal`, or `private`. Do not assume `@testable import` is a skeleton key.

---

### 2. DPM-Solver Sigma Threshold Is ~0.17, Not < 0.1

**What happened:** Sortie 2's `lastSigmaApproachesZero` test assumed the last sigma for a 20-step schedule would be < 0.1. The actual value is ~0.17. The test would have failed.

**What was built to handle it:** The agent read the actual sigma values after `configure(numSteps: 20)` and adapted the assertion to `sigmas.first > 10 * sigmas.last`, which passes for any reasonable schedule without hard-coding the threshold.

**Should we have known this?** Yes. The plan guessed at a threshold without running the scheduler. A five-line exploratory script would have revealed the real value.

**Carry forward:** Do not write threshold assertions for scheduler math without first running the scheduler with the test configuration and observing the actual output range.

---

### 3. DPM-Solver `step()` Does Not Guarantee L2 Norm Reduction

**What happened:** Sortie 2 was asked to write `stepReducesNoiseMagnitude` asserting that `step()` output has lower L2 norm than the noisy input. DPM-Solver is a reconstruction formula — it moves toward the predicted clean image but does not monotonically reduce L2 norm per step.

**What was built to handle it:** The test was renamed and rewritten to verify that `step()` produces finite output that differs from input — which is verifiable regardless of the optimization landscape.

**Should we have known this?** Yes. The assertion was written based on a mental model of denoising as noise reduction. Reading the DPM-Solver paper (or even the docstring) would have shown this is a predictor-corrector, not a simple noise subtractor.

**Carry forward:** Scheduler step tests should verify structural properties (shape, finiteness, non-identity) rather than assuming directional properties (norm reduction, monotone decrease) without algorithm-specific justification.

---

### 4. `DiffusionPipeline` Components Are Private — No Clean Test Loading Path

**What happened:** `DiffusionPipeline.generate()` now guards `encoder.isLoaded`, `backbone.isLoaded`, `decoder.isLoaded`. The smoke tests in `PipelineAssemblyTests` construct a pipeline and call `generate()` directly. The internal components (`encoder`, `backbone`, `decoder`) are private stored properties. There is no public API to load them without real weight files registered with Acervo. `loadModels(progress:)` silently skips all calls when `allComponentIds: []`.

**What was built to handle it:** `internal func loadSyntheticWeights()` was added to `DiffusionPipeline.swift`. It calls `apply(weights: ModuleParameters(parameters: [:]))` on all three components. Mock components accept empty weights and set `isLoaded = true`. The 4 failing smoke tests now call `try await pipeline.loadSyntheticWeights()` before `generate()`.

**Should we have known this?** Yes. The EXECUTION_PLAN.md noted that the `guard isLoaded` change was added to `DiffusionPipeline.swift` pre-mission. Anyone reading that file would have seen the smoke tests immediately break. The plan did not account for this.

**Carry forward:** When a production change to a public method adds entry guards, immediately search for all existing call sites in the test suite and update them in the same sortie. Do not treat this as a separate task.

---

### 5. Scheme Name Is `SwiftTuberia-Package`, Not `SwiftTuberia`

**What happened:** The EXECUTION_PLAN.md exit criteria specified `-scheme SwiftTuberia` in every xcodebuild command. The actual scheme is `SwiftTuberia-Package`. Every agent had to discover and correct this.

**What was built to handle it:** Each agent ran `xcodebuild -list` or received the correction in its prompt (after the supervisor discovered it mid-mission) and used `SwiftTuberia-Package`.

**Should we have known this?** Yes. Running `xcodebuild -list` before writing the plan would have taken 10 seconds.

**Carry forward:** Always run `xcodebuild -list` and record the exact scheme names in the EXECUTION_PLAN.md before writing any exit criteria.

---

### 6. Parallel Dispatch Caused a Transient Compile Race

**What happened:** Sorties 2 (TuberiaCatalogTests) and 7 (TuberiaGPUTests) were dispatched simultaneously. Sortie 7 was still writing `PipelineErrorTests.swift` when Sortie 2's `xcodebuild` compiled the full scheme. `xcodebuild` picked up a partially-written file, got a compile error, and the Sortie 2 agent worked around it with `-skip-testing TuberiaGPUTests`.

**What was built to handle it:** The Sortie 2 agent used `-skip-testing TuberiaGPUTests`. The file was complete and correct by the time Sortie 2 finished. No data was lost.

**Should we have known this?** This is a fundamental risk of parallel dispatch when agents write to files in targets that other agents compile. It was not anticipated.

**Carry forward:** When dispatching sorties in parallel across different test targets, note that xcodebuild compiles the full scheme by default. Either (a) use `-skip-testing <OtherTarget>` defensively in exit criteria, or (b) accept that transient race errors may appear and will self-resolve.

---

## Section 2: Process Discoveries

### What the Agents Did Right

#### 1. Source-First Test Writing

Every agent read the relevant source files before writing assertions. Sortie 3's agent verified the exact transposition axes in `SDXLVAEDecoder.tensorTransform` before asserting output shapes. Sortie 6's agent read `LoRALoader.swift` key-matching logic and correctly identified that `unet.` prefix stripping was aspirational — the implementation didn't support it — and documented the actual behavior instead of asserting imaginary behavior.

**Carry forward:** This is the correct pattern. Preserve it. Include an explicit "read the source first" step in every sortie that writes assertions about internal behavior.

#### 2. Opus Model Was Right for Sortie 6

The LoRA key-convention sortie required deep reading of underdocumented string-matching logic, then writing tests against that logic without misrepresenting it. The opus model's careful approach of distinguishing "what the code does" from "what the spec says it should do" saved the mission from 4 incorrect tests.

**Carry forward:** When a sortie requires inferring contracts from implementation rather than documentation, start with sonnet and consider opus if the logic is tangled.

#### 3. `eval(result)` Over `eval()`

The plan exit criteria used `grep -c 'eval()'` expecting MLX array evaluation. MLX's `eval` requires an argument — `eval()` is not valid Swift. Agents wrote `eval(result)`, `eval(embeddingWeight)`, etc. The Sortie 3 agent explicitly flagged this discrepancy.

**Carry forward:** Update exit criteria to use `grep -c 'eval('` (no closing paren) to match `eval(anyVar)`.

---

### What the Agents Did Wrong

#### 1. Sortie 1: Private Property Access Without Source Inspection

The Sortie 1 agent wrote an assertion accessing `encoder.transformer.embedding.weight` without first checking the visibility of `transformer`. This caused a SourceKit diagnostic before xcodebuild ran. The agent self-corrected, but the error was avoidable with 30 seconds of source reading.

**Carry forward:** Add an explicit task to every sortie that accesses component internals: "Before writing any assertion, grep for `private var` / `internal var` in the target source file and confirm the property you plan to access is at least `internal`."

---

### What the Planner Did Wrong

#### 1. Exit Criteria Used Wrong Scheme Name

`-scheme SwiftTuberia` appears throughout the EXECUTION_PLAN.md exit criteria. The actual scheme is `SwiftTuberia-Package`. Every agent discovered this independently.

#### 2. Scheduler Test Assertions Were Guesses

`lastSigmaApproachesZero` (threshold < 0.1) and `stepReducesNoiseMagnitude` (L2 norm reduction) were written without running the code. Both were wrong.

#### 3. PipelineAssemblyTests Breakage Was Not Anticipated

The plan noted the `DiffusionPipeline.generate()` guard change as resolved, but did not assess whether existing tests calling `generate()` would break. A one-minute grep (`grep -rn 'pipeline.generate\|\.generate(' Tests/TuberiaGPUTests/`) before writing the plan would have flagged the 4 smoke tests.

#### 4. No Commit Strategy

The mission plan had no sortie or instruction for committing the work. All changes remain uncommitted on the mission branch. The brief is being written against a dirty working tree.

---

## Section 3: Open Decisions

### 1. Should `loadSyntheticWeights()` Stay in Production Code?

**Why it matters:** It's marked `internal` and only exists to support tests. It's a testing hook in production code, which creates a category confusion.

**Options:**
- **A.** Keep it as-is (`internal` in `DiffusionPipeline.swift`). Simple, works.
- **B.** Move it to a test-only extension in a file conditionally compiled with `#if DEBUG` or `#if TESTING`.
- **C.** Refactor the smoke tests to use `MockPipelineRecipe` with an `autoLoad: Bool` option that calls `apply(weights:)` automatically on all components during `DiffusionPipeline.init`.

**Recommendation:** Option C is the cleanest — it removes the hook from production code entirely and makes the mock recipe self-documenting. But it requires a small refactor of `MockPipelineRecipe`. Option A is acceptable for now if the hook is clearly documented.

---

### 2. Should `unet.` Prefix Stripping Be Implemented?

**Why it matters:** The `unetPrefixIsStripped` test documents that adapter keys with a `unet.` prefix will **not** match base model keys. This means real LoRA adapters targeting UNet layers via the `unet.` convention will silently fail to merge.

**Options:**
- **A.** Implement prefix stripping in `LoRALoader.parseLoRAKey`.
- **B.** Accept current behavior; document that `unet.`-prefixed adapters are not supported.

**Recommendation:** Check whether any real adapters you intend to load use `unet.` prefix keys. If yes, Option A is required. If not, Option B with documentation is fine.

---

### 3. Who Commits the Work and How?

**Why it matters:** All mission changes are uncommitted. If the branch is switched or the workspace is cleaned before committing, the work is lost.

**Options:**
- **A.** Commit everything now as one large commit.
- **B.** Commit per logical group (test deletions, new test files, production changes) for clean history.

**Recommendation:** Option B. Three commits minimum: (1) test deletions and strengthenings, (2) new test files, (3) `DiffusionPipeline.swift` changes (`loadSyntheticWeights` + `guard isLoaded`).

---

## Section 4: Sortie Accuracy

| Sortie | Task | Model | Attempts | Accurate? | Notes |
|--------|------|-------|----------|-----------|-------|
| 1 | Remove 4 redundant tests; strengthen `applyWeightsDoesNotCrash` | sonnet | 1 (self-corrected) | Mostly | Accessed `private` property first; corrected before xcodebuild. No retry needed. |
| 2 | DPMSolver + FlowMatchEuler scheduler CPU tests | sonnet | 1 | Mostly | 2 of 9 assertions were wrong assumptions; agent adapted correctly. Race with Sortie 7 caused transient compile error; worked around with `-skip-testing`. |
| 3 | SDXLVAEDecoder tensor-transform tests | sonnet | 1 | Yes | Read source first, derived correct transposition axes, wrote accurate tests. Flagged `eval(result)` vs `eval()` discrepancy correctly. |
| 4 | ImageRenderer CPU tests | haiku | 1 | Yes | Clean first attempt. Appropriate use of haiku — simple, well-scoped task. |
| 5 | Remove 2 LoRA tests; add device capability tests | haiku | 1 | Yes | Clean first attempt. `hasNeuralAccelerators` name (vs `hasNeuralEngine` in original requirements) was already corrected in plan. |
| 6 | LoRAKeyConventionTests suite | opus | 1 | Yes | Correctly identified `unetPrefixIsStripped` tests aspirational behavior. Documented actual implementation behavior rather than asserting fiction. High-value sortie. |
| 7 | PipelineErrorTests thrown-condition tests | sonnet | 1 | Partial | New tests pass. PipelineAssemblyTests breakage was out of scope but should have been in scope — the `generate()` guard change was known pre-mission. |
| fix | PipelineAssemblyTests 4 smoke tests | sonnet | 1 | Yes | Added `loadSyntheticWeights()`. All 42 TuberiaGPUTests pass. Clean execution. |

---

## Section 5: Harvest Summary

The mission's core work was accurate: 26 new tests across three targets, 6 redundant tests deleted, all suites passing. The gaps came from plan assumptions that were never verified against the actual codebase — wrong scheme name, wrong sigma threshold, wrong L2 norm expectation, and an unexamined call-site impact when `generate()` guards were added. The `private` vs `internal` distinction bit Sortie 1. The single most important thing for the next iteration: **run the code before writing assertions about it**. Every hard discovery in this brief was avoidable with 10 minutes of exploratory grepping or scripting before the plan was finalized.

---

## Section 6: Files

**Preserve (read-only reference):**

| File | Branch | Why |
|------|--------|-----|
| `Tests/TuberiaCatalogTests/DPMSolverSchedulerTests.swift` | `mission/blunt-blade-reforge/01` | 9 new scheduler CPU tests |
| `Tests/TuberiaCatalogTests/FlowMatchEulerSchedulerTests.swift` | `mission/blunt-blade-reforge/01` | 7 new scheduler CPU tests |
| `Tests/TuberiaCatalogTests/ImageRendererUnitTests.swift` | `mission/blunt-blade-reforge/01` | 6 new CoreGraphics pixel tests |
| `Tests/TuberiaGPUTests/ContractTests/PipelineErrorTests.swift` | `mission/blunt-blade-reforge/01` | 3 new `#expect(throws:)` tests for generate() guards |
| `Sources/Tuberia/Pipeline/DiffusionPipeline.swift` | `mission/blunt-blade-reforge/01` | `guard isLoaded` checks + `loadSyntheticWeights()` testing hook |

**Discard (safe to lose on rollback):**

| File | Why it's safe to lose |
|------|----------------------|
| `SUPERVISOR_STATE.md` | Mission state only; will be regenerated on next start |
| `OPERATION_BLUNT_BLADE_REFORGE_01_BRIEF.md` | Will be archived to `Docs/complete/` before rollback |

---

## Section 7: Iteration Metadata

**Starting point commit:** `c3840ebf54379f10ff21554938ea67eb7b470dee` (`ci: add workflow_dispatch trigger to allow manual test runs`)
**Mission branch:** `mission/blunt-blade-reforge/01`
**Final commit on mission branch:** `c3840ebf` (all mission changes are uncommitted — commit before any rollback)
**Rollback target:** `c3840ebf54379f10ff21554938ea67eb7b470dee`
**Next iteration branch:** `mission/blunt-blade-reforge/02`
