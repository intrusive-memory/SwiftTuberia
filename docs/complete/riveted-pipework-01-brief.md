# Iteration 01 Brief — OPERATION RIVETED PIPEWORK

> **Terminology**: A *mission* is a definable scope of work. A *sortie* is one atomic agent task within it. A *brief* is this post-mission debrief.

**Mission:** Complete SwiftTuberia's migration from SwiftAcervo v1 partial-registration integration to complete v2 abstracted-access integration with SHA-256 integrity verification and standardized CDN workflows.
**Branch:** `mission/riveted-pipework/01`
**Starting Point Commit:** `dcc1eec` (refactor: reorganize requirements and simplify catalog registration)
**Final Commit:** `de072ec`
**Sorties Planned:** 8
**Sorties Completed:** 8
**Sorties Failed/Blocked:** 0
**Retries:** 0 (all single-attempt)
**Duration:** ~2.2 hours of agent wall time (excluding supervisor orchestration)
**Outcome:** Complete
**Verdict:** **Keep the code and merge forward.** All 8 commits are mergeable. The remaining caveats (CDN not yet populated, pre-existing Metal GPU crash) are out-of-scope issues that do not invalidate the mission's work.

---

## Section 1: Hard Discoveries

### 1. T5 model is sharded, not monolithic

**What happened:** Plan assumed `intrusive-memory_t5-xxl-int4-mlx` has a single `model.safetensors` file (4 total files with config/tokenizer). Reality: T5 is split across `model-00000-of-00005.safetensors` through `model-00004-of-00005.safetensors` plus 4 metadata/tokenizer files = 9 T5 files, plus 2 VAE files = 11 total ComponentFile entries. S2 discovered this by actually listing the artifact set.
**What was built to handle it:** S2 populated all 11 entries correctly; S8 corrected REQUIREMENTS.md's "6 entries" claim with an explicit note about T5 sharding. The exit criterion `grep -c 'sha256:' == 6` was effectively replaced by the intent check (`grep -v 'sha256:' ... wc -l == 0`).
**Should we have known this?** Yes — a one-line `ls` on the CDN slug or the model repo would have revealed sharding. The plan author accepted the T4 audit finding at face value without verifying artifact structure.
**Carry forward:** Before writing plan exit criteria with hard-coded counts, list the actual artifact set. Exit criteria should express *intent* (every entry has sha256), not *count* (entries == 6).

### 2. CDN is empty at mission time

**What happened:** S2 attempted to pull canonical artifacts from `https://pub-8e049ed02be340cbb18f921765fd24f3.r2.dev` per the `ensure-model-cdn.yml` slugs. Every URL returned 404. Plan assumed CDN was populated.
**What was built to handle it:** S2 sourced SHA-256 digests from the local group container `group.intrusive-memory.models/SharedModels` (the path SwiftAcervo writes downloaded artifacts to). S7's VerifyComponentManifest tool was designed specifically to surface drift between CatalogRegistration.swift and an uploaded manifest — its first run in CI will *fail* until real artifacts are pushed matching these local checksums.
**Should we have known this?** Yes — checking `curl -I` against one CDN URL before writing the plan would have revealed the state.
**Carry forward:** This mission's checksums are only valid if the first CDN push uses byte-for-byte identical artifacts from the local group container. Any re-quantization, re-export, or different model source at CDN-push time will break S7's gate. Document this as a precondition for the first CI run.

### 3. Swift 6 strict concurrency forces `@escaping` on progress closures crossing `@Sendable` boundaries

**What happened:** S3 added `try await Acervo.ensureComponentReady(componentId, progress: …)` inside `loadModels(progress:)`. The existing `progress` parameter was non-escaping. Swift 6 refused to let the non-escaping closure be captured inside the `@Sendable` escaping closure that `ensureComponentReady` expects. S3 added `@escaping` to the `progress` parameter and updated the `GenerationPipeline` protocol to match.
**What was built to handle it:** `loadModels(progress:)` now takes `@escaping (Double, String) -> Void`. Not a source-breaking change — all callers use trailing closure syntax — but an API signature ripple.
**Should we have known this?** Partial yes. The plan called for threading progress through `ensureComponentReady`; the concurrency implication was not obvious until the build failed.
**Carry forward:** When a plan task says "thread progress through X", the plan reviewer should predict `@escaping` changes. Ripple effects on protocol conformance need an explicit "update conformers" task.

### 4. `PipelineError.insufficientMemory` already existed in the exact form S4 was meant to add

**What happened:** S4's plan task #2 was "Define a pipeline-specific error variant `PipelineError.insufficientMemory(required: UInt64, available: UInt64)` in `Sources/Tuberia/Pipeline/PipelineError.swift`". The case already existed with that exact signature — the plan author wrote the task without grepping for the case.
**What was built to handle it:** S4 skipped the "add" step, just wired the existing case into the call site. Saved ~5 minutes of work.
**Should we have known this?** Yes — `grep -n 'insufficientMemory' Sources/Tuberia/Pipeline/PipelineError.swift` takes one second.
**Carry forward:** Plan tasks that "define X" should be preceded by a `grep` check to see if X already exists. Or state them as "ensure X exists" with a check-first-add-if-missing pattern.

### 5. Pre-existing Metal GPU crash in `T5EncodeWithSyntheticWeightsTests.encodeOutputIsNonZero`

**What happened:** S6 ran the full `xcodebuild test` suite and the run crashed on a Metal GPU operation in the T5 encoder test. S6 stashed its changes, re-ran, reproduced the crash on HEAD — confirmed pre-existing, not caused by this mission.
**What was built to handle it:** Nothing — out of scope. S8 used `-skip-testing:TuberiaGPUTests` to validate its doc-only changes.
**Should we have known this?** Impossible to know at plan time unless we ran the full suite on the starting-point commit. The plan's Master Acceptance DoD item #2 (`xcodebuild test passes on a clean clone`) was written without verifying this premise.
**Carry forward:** **This mission does NOT satisfy Master Acceptance DoD item #2 and cannot, without a separate T5 GPU fix.** File as a new mission. Consider amending the DoD to allow a named skip-list until the crash is fixed.

### 6. `@testable import SwiftAcervo` from TuberiaCatalogTests reaches internal overloads

**What happened:** S6 needed the internal `withComponentAccess(_:in:perform:)` overload. Plan noted it was "available at `AcervoManager.swift:455`" but didn't specify the access path. S6 discovered `@testable import SwiftAcervo` works because TuberiaCatalogTests is compiled with `-enable-testing` and SwiftAcervo is a transitive dependency.
**What was built to handle it:** S6's test files use `@testable import SwiftAcervo`. Avoids mutating global `Acervo.customBaseDirectory` and eliminates a race condition that would have caused flaky tests.
**Should we have known this?** Yes — this is a Swift testing pattern, not a surprise. Plan should have said "access via @testable import" explicitly.
**Carry forward:** When a plan task requires internal API access, state the access mechanism (`@testable`, `@_spi(Internal)`, friend-module, etc.) — don't leave it to the agent to figure out.

---

## Section 2: Process Discoveries

### What the Agents Did Right

#### 1. Protocol-backed seam pattern adopted consistently

**What happened:** S3 introduced `ComponentReadinessService` as a protocol-backed seam so tests could inject a double. S4 observed the pattern and mirrored it with `memoryGate: @Sendable (UInt64) async throws -> Void` for the memory-validation seam.
**Right or wrong?** Right. Consistency within the codebase makes the pipeline's testable seams uniform.
**Evidence:** Two sorties on separate dispatches converged on the same pattern without the supervisor prompting it.
**Carry forward:** Future sorties in this codebase can assume "inject via closure or protocol property" as the seam idiom.

#### 2. MLX.save used for synthetic safetensors instead of manual byte construction

**What happened:** S6 needed tiny valid `.safetensors` files for integration tests. Instead of reading the safetensors header spec and constructing bytes, the agent called `MLX.save(arrays:url:)`.
**Right or wrong?** Right. Leveraged existing toolchain; test is robust to any format changes upstream.
**Evidence:** S6 tests ran clean on first xcodebuild invocation.
**Carry forward:** Prefer library-generated fixtures over hand-crafted bytes when the library is already a dependency.

#### 3. S7's scratch-manifest divergence test

**What happened:** Plan's exit criterion suggested "edit one hex digit in CatalogRegistration.swift, verify verifier catches it, revert". S7 instead wrote a scratch `manifest.json` with intentionally wrong values and ran the verifier against it — never touched source.
**Right or wrong?** Right. Safer, shorter, and doesn't risk a failed-revert leaving mutated source in the commit.
**Evidence:** `git diff dc88d6d -- Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` showed zero output at S7 commit time.
**Carry forward:** When a plan suggests "edit source, verify, revert", check whether an out-of-band test fixture achieves the same verification. Usually it does.

#### 4. S8 caught two plan defects

**What happened:** S8 identified that the Master Acceptance grep `^\| *[6-9] \|.*✅ DONE` only matches single-digit rows (6–9), not the two-digit rows 10–12. S8 also corrected the ComponentFile count from 6 → 11 in REQUIREMENTS.md.
**Right or wrong?** Right. Named the plan bug instead of silently adapting.
**Evidence:** S8's final report called out the regex issue explicitly.
**Carry forward:** Agents that notice plan defects should flag them, not route around them silently.

### What the Agents Did Wrong

#### 1. No agent pushed back on the "6 entries" number before S2 started

**What happened:** S1, S2, S3 all had read access to CatalogRegistration.swift at dispatch. None questioned whether "6 ComponentFile entries" was accurate. S2 discovered reality only after running `curl` on the CDN URLs.
**Right or wrong?** Mildly wrong. A 5-second `grep -c ComponentFile Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` would have caught it before writing any code.
**Evidence:** S2's report notes the deviation after-the-fact, not before starting.
**Carry forward:** Agent prompts should include a "sanity-check the plan's assumed counts/file lists before starting" instruction.

### What the Planner Did Wrong

#### 1. Plan did not test its own exit-criterion regex patterns

**What happened:** Two exit criteria used regex patterns that didn't match reality:
- S2's `grep -c 'sha256: ...' == 6` — wrong because count is 11, not 6.
- Master Acceptance `^\| *[6-9] \|.*✅ DONE == 7` — wrong because `[6-9]` is a single-digit class and rows 10, 11, 12 exist.
**Right or wrong?** Wrong. Exit criteria ARE specifications; unrun specs drift from reality.
**Evidence:** S2 and S8 both had to work around incorrect regex assumptions.
**Carry forward:** Refinement Pass 1 (atomicity/testability) should actually *execute* every grep-based exit criterion against the starting-point working tree to confirm it returns a sensible value. Any criterion that cannot be pre-verified must be flagged as a risk.

#### 2. Plan did not verify CDN was populated

**What happened:** Plan's S2 task #1 says "Pull canonical MLX artifacts from the CDN for each component". It named slugs but didn't verify the slugs resolved. S2 got 404s.
**Right or wrong?** Wrong. CDN state is external and should be probed before planning depends on it.
**Evidence:** S2 fell back to a local group container, producing checksums that may not match the eventual CDN push.
**Carry forward:** When a task depends on external infrastructure (CDN, API, database fixture), add a pre-plan verification step. If the external state is not ready, either add a mission to prepare it, or scope the plan to explicitly work from local sources.

#### 3. Plan had task that duplicated existing code

**What happened:** S4's task #2 called for *defining* `PipelineError.insufficientMemory`, which already existed. Plan author did not grep before writing the task.
**Right or wrong?** Wrong, but self-correcting — S4's agent discovered and skipped.
**Evidence:** S4 report says "PipelineError.insufficientMemory already existed in PipelineError.swift with the correct signature — no change needed".
**Carry forward:** Refinement Pass 1 should include "grep for every symbol this task claims to introduce; mark as 'add' or 'wire' accordingly".

#### 4. Master Acceptance assumed a green test suite at starting-point

**What happened:** DoD item #2 (`xcodebuild test passes on a clean clone`) was impossible to satisfy because of a pre-existing Metal GPU crash that predated this mission.
**Right or wrong?** Wrong premise, not wrong criterion. The criterion *should* be green — it just can't be until a separate bug is fixed.
**Evidence:** S6 stashed changes and reproduced the crash on `dcc1eec`.
**Carry forward:** Before stating a DoD check that requires a green baseline, run the check on the starting-point commit and document its state. If red, either (a) include the fix in the mission, (b) scope the mission around the failing area, or (c) use a pinned skip-list.

### What the Supervisor Did Wrong

#### 1. Dispatched S5 while S2 was still running — violated Layer 2 gate

**What happened:** Plan explicitly sequences Layer 2 AFTER Layer 1. S2 is Layer 1. S5 is Layer 2. Supervisor dispatched S5 immediately after S3 completed, ignoring that S2 was still in flight.
**Right or wrong?** Wrong. S2's recovery revert of `DiffusionPipeline.swift` destroyed S5's uncommitted edits. S5's agent re-applied them from memory and completed successfully — pure luck.
**Evidence:** S2's report explicitly says "working tree had partial S5 (`REQ-PIPE-03`) changes applied to DiffusionPipeline.swift that broke the build… the file was reverted to HEAD".
**Carry forward:** Respect the plan's layer gates, not just its "depends on" edges. Layer gates encode "don't share a filesystem with other agents in a lower layer". For the next iteration: either (a) enforce layer gates in SUPERVISOR_STATE.md, or (b) use `Agent({ isolation: "worktree" })` for every parallel dispatch.

#### 2. Did not use git worktree isolation for parallel dispatches

**What happened:** All three Layer 1 sorties (S1/S2/S3) ran against the same filesystem. S1/S3 touched disjoint files and were fine. S2's xcodebuild-resolve + S5's concurrent edits would have collided even without the layer-gate violation.
**Right or wrong?** Wrong. The `isolation: "worktree"` parameter exists in the Agent tool for exactly this case.
**Evidence:** The Agent tool doc lists `isolation: "worktree"` as a first-class parameter. I did not use it on any dispatch.
**Carry forward:** Default to `isolation: "worktree"` for any concurrent sortie dispatch. Only serialize-without-worktree if sorties are guaranteed to modify disjoint files AND not invoke xcodebuild concurrently.

---

## Section 3: Open Decisions

### 1. Does the first CI run succeed or fail S7's manifest verifier?

**Why it matters:** S2 sourced SHA-256 digests from the local group container. If the first CDN push uses the same byte-identical artifacts, S7 passes. If CI auto-generates or re-downloads artifacts from a different source (HuggingFace, re-quantization), S7 fails and blocks the `ensure-model-cdn.yml` workflow.
**Options:**
- A: Pre-stage the CDN with the exact local artifacts before the next CI run. Verify byte-match manually.
- B: Treat the first CI run as the source-of-truth moment, update CatalogRegistration.swift to match whatever CI produces, then re-commit. Accept that this mission's checksums are provisional.
- C: Add a one-shot "prime the CDN" workflow that uploads the local artifacts and records the manifest, then CatalogRegistration.swift is verified against *that* manifest.
**Recommendation:** C. It makes the provisioning explicit and auditable. Option B is a slippery slope — it says "checksums are whatever CI says today", defeating the purpose of integrity verification.

### 2. Is the pre-existing T5 Metal GPU crash in-scope for v2 completion?

**Why it matters:** Master Acceptance DoD item #2 cannot pass until this is fixed. Ignoring it means the mission is "complete" by 7/8 DoD items but not by the strict letter.
**Options:**
- A: File a new mission (`OPERATION <SOMETHING> T5-GPU`) and pretend DoD #2 is satisfied "modulo pre-existing bug".
- B: Add the fix to this mission as S9 and re-open it.
- C: Amend the DoD to allow a pinned skip-list.
**Recommendation:** A. The crash is in GPU/Metal code, orthogonal to the v2 integration work. Bundling would muddy both scopes. But amend this iteration's final status to "complete with known-failing GPU test" and reference the new mission.

### 3. Should uncommitted supervisor artifacts land on the branch?

**Why it matters:** Currently `EXECUTION_PLAN.md` has my frontmatter edits (feature_name, mission_branch, starting_point_commit, iteration) uncommitted, and `SUPERVISOR_STATE.md` is untracked. If the branch merges into development, neither set of edits goes with it.
**Options:**
- A: Commit `EXECUTION_PLAN.md` frontmatter (useful for brief.md iteration detection in future runs). Do not commit `SUPERVISOR_STATE.md` (ephemeral).
- B: Commit both.
- C: Commit neither (drop the frontmatter on rollback, never merge state).
**Recommendation:** A. The frontmatter enables iteration counting; the state file is scaffolding. The brief itself (archived to `docs/complete/`) is the permanent record.

### 4. Worktree isolation policy for future missions

**Why it matters:** Without a durable rule, the supervisor may repeat the same S5/S2 collision.
**Options:**
- A: Mandate `isolation: "worktree"` on every parallel sortie dispatch.
- B: Mandate serialization for any sortie that runs `xcodebuild`.
- C: Both — worktrees for filesystem isolation, plus serialized xcodebuild to avoid DerivedData contention.
**Recommendation:** C. Worktree handles the source-of-truth collision; xcodebuild serialization handles the build-cache issue. The skill.md could codify this.

---

## Section 4: Sortie Accuracy

| Sortie | ID | Model | Attempts | Accurate? | Notes |
|--------|------|-------|----------|-----------|-------|
| S1 | REQ-T5 | haiku | 1 | ✅ | Tightest sortie — single-version bump + one changelog entry. Haiku was right call. |
| S2 | REQ-T4 | sonnet | 1 | ✅ with deviation | Corrected 6→11 entries against a wrong plan; CDN fallback to local source preserved forward progress. |
| S3 | REQ-PIPE-01 | sonnet | 1 | ✅ | Established the ComponentReadinessService seam pattern that S4 reused. |
| S4 | REQ-PIPE-02 | sonnet | 1 | ✅ | Skipped plan's duplicate "define PipelineError.insufficientMemory" step because case existed. Net positive. |
| S5 | REQ-PIPE-03 | sonnet | 1* | ✅ with caveat | Midflight working-tree edits destroyed by S2's recovery revert. Agent re-applied from memory and completed. Should not be repeated. *counted as 1 attempt because the agent never exited — it pushed through. |
| S6 | REQ-INT-01 | sonnet | 1 | ✅ | Longest-running sortie (63 min). MLX.save approach + @testable import were both smart calls. |
| S7 | REQ-CDN-01 | sonnet | 1 | ✅ | Fastest non-haiku sortie (4 min). Scratch-manifest divergence test was safer than plan's edit-and-revert. |
| S8 | REQ-DOC-01 | sonnet | 1 | ✅ | Caught plan's regex bug; corrected ComponentFile count. Verified 13 `[x]` items. |

**Overall accuracy**: 8/8 first-attempt. No retries. No backoffs. No FATAL states.

---

## Section 5: Harvest Summary

The v2 integration is structurally complete — every call site, every seam, every test exists on `mission/riveted-pipework/01`. The single most important thing I now know that I didn't before: **this codebase's external state (CDN, test baseline) was not verified before the plan was written, and two of the six plan defects (exit-criterion regex, `[6-9]` bug) were purely the result of skipping Refinement Pass 1's promised "execute the check against the starting state" step.** For the next iteration of any mission in this repo, Refinement Pass 1 must actually run every machine-verifiable exit criterion against HEAD and record the observed value — not merely sanity-read them.

Secondary finding: the Agent tool's `isolation: "worktree"` parameter is a free win for parallel sortie dispatches. Not using it on this mission cost a near-miss on S5's uncommitted work.

---

## Section 6: Files

### Preserve (read-only reference for next iteration)

| File | Branch | Why |
|------|--------|-----|
| `OPERATION_RIVETED_PIPEWORK_01_BRIEF.md` → `docs/complete/riveted-pipework-01-brief.md` | any (archived) | This brief. Carry-forward facts for iteration 02 or successor missions. |
| `EXECUTION_PLAN.md` (frontmatter added) | `mission/riveted-pipework/01` | The plan as executed. Frontmatter records iteration metadata for detection. |
| All 8 sortie commits | `mission/riveted-pipework/01` | Merge-ready. |

### Discard (safe to lose — not merged anywhere)

| File | Why it's safe to lose |
|------|----------------------|
| `SUPERVISOR_STATE.md` | Ephemeral scaffolding; everything non-ephemeral is in this brief. |
| `default.profraw` | Xcode coverage artifact from a test run. Regenerated on next test. |

---

## Section 7: Iteration Metadata

**Starting point commit:** `dcc1eec` (refactor: reorganize requirements and simplify catalog registration)
**Mission branch:** `mission/riveted-pipework/01`
**Final commit on mission branch:** `de072ec`
**Rollback target (if discarding):** `dcc1eec`
**Next iteration branch (if iterating):** `mission/riveted-pipework/02` (but see Verdict — this mission is keep, not iterate)

---

## Recommendation to the User

**Do not roll back.** The verdict is "keep the code". Specifically:

1. Commit `EXECUTION_PLAN.md` frontmatter on `mission/riveted-pipework/01` (Open Decision 3 → Option A).
2. Archive this brief to `docs/complete/riveted-pipework-01-brief.md` and remove `SUPERVISOR_STATE.md` from the workspace (cleanup per skill).
3. Merge `mission/riveted-pipework/01` into `development` after eyeballing the 8 commits.
4. File a follow-up mission for the T5 Metal GPU crash (Open Decision 2 → Option A).
5. Address the CDN prime (Open Decision 1 → Option C) before relying on `ensure-model-cdn.yml` in CI.

The rollback ritual in `brief.md` is for "discard and iterate". It does not apply here.
