# Iteration 02 Brief — OPERATION VANISHING MANIFEST

> **Terminology:** *Mission* = the definable scope of work. *Sortie* = one atomic agent dispatch within that mission. *Brief* = post-mission review.

> **Numbering note:** EXECUTION_PLAN.md frontmatter is `iteration: 2`. Filename uses `03` to avoid colliding with `docs/complete/vanishing-manifest-02-brief.md` (the iteration-01 final brief, which itself used `02` to avoid colliding with the iteration-01 interim brief). The filename NN has effectively become a monotonic file counter; the iteration number lives in the content.

**Mission:** Make SwiftTuberia never deal with files — Acervo provides all model access (V2-compliance requirements across SwiftAcervo + SwiftTuberia).
**Branch:** `mission/vanishing-manifest/02`
**Starting Point Commit:** `36887b9` (deps: migrate tokenizer package — swift-transformers → swift-tokenizers)
**Sorties Planned:** 6 (5 required + 1 optional)
**Sorties Completed:** 5 of 5 required (Acervo-S1, Acervo-S2, Tuberia-S1, Tuberia-S2, Tuberia-S3); Acervo-S3 (optional) skipped
**Sorties Failed/Blocked:** 0 — all five completed on first attempt, zero PARTIAL/BACKOFF/FATAL
**Duration:** ~1 working session, 2026-04-27. Cost-relative: 5 sonnet sorties. Five total dispatches, zero retries.
**Outcome:** **Complete (Definition of Done met)**
**Verdict:** **Land the work, don't roll back.** All six DoD checks pass, the test suite passes (29 tests in SwiftTuberia, 470 in SwiftAcervo), v0.8.3 is published. The right next step is a PR `mission/vanishing-manifest/02 → main`, not the rollback ritual. The carry-forward rules from iteration 01 (per-sortie commit gate U1, sonnet-floor on release sortie, U4 auth precondition) paid for themselves on first contact.

---

## Section 1: Hard Discoveries

### 1. `xcodebuild -resolvePackageDependencies` does not bump pinned SPM versions

**What happened:** Tuberia-S3 Task 1 stipulated `xcodebuild -resolvePackageDependencies` to pull SwiftAcervo ≥ 0.8.3. The agent ran it. `Package.resolved` stayed pinned at `0.8.2`. The fix was `swift package update SwiftAcervo`, which fetched the v0.8.3 tag and updated the pin correctly.
**What was built to handle it:** Nothing structural — the agent pivoted at runtime. The pivot was defensible: CLAUDE.md only forbids `swift build` and `swift test`, not `swift package update`. But the plan specified the wrong tool.
**Should we have known this?** Yes. `-resolvePackageDependencies` resolves *missing or invalid* dependencies; it does not upgrade pins. To upgrade an existing pin, you need `swift package update <name>` or you delete the pin first (or delete `Package.resolved` entirely and re-resolve from scratch). This is a documented SwiftPM behavior, not a bug.
**Carry forward:** When a plan needs to *bump* a remote SPM dependency to a newly-released version, the command is `swift package update <name>`, NOT `xcodebuild -resolvePackageDependencies`. The auto-fix table in this iteration's plan explicitly *replaced* "swift package update" with "xcodebuild -resolvePackageDependencies" during refinement — that was an over-correction. Refinement should preserve `swift package update` when the goal is a version bump.

### 2. `Package.resolved` was gitignored on this project

**What happened:** Tuberia-S3 needed to commit `Package.resolved` to lock the v0.8.3 pin. The plan listed it under "Touched files." But `.gitignore` had `Package.resolved` listed. The agent used `git add -f` to force-track it. The plan's exit criteria didn't anticipate this.
**What was built to handle it:** Force-add. The file is now tracked. Future edits will appear in `git status` (gitignore is bypassed for already-tracked files), but the gitignore intent is now incoherent — the file claims it should be ignored but is actually tracked.
**Should we have known this?** Yes — by reading `.gitignore` during refinement.
**Carry forward:** Refinement passes (especially Pass 4 — open questions) must include a `.gitignore` audit when the plan stipulates committing files. If a "touched file" is gitignored, the planner has to make a deliberate decision: (A) untrack the gitignore entry, (B) skip the commit and lock the version another way (Package.swift constraint), or (C) flag for the user to decide before execution.

### 3. SwiftAcervo had two version-bearing files; one had drifted

**What happened:** Acervo-S2's release agent found that `Sources/SwiftAcervo/Acervo.swift` (`public static let version`) was at `0.8.2` as expected, but `Sources/acervo/Version.swift` (`let acervoVersion`) was at `0.8.0` — drifted across two prior releases (0.8.1, 0.8.2 didn't bump the CLI version). Both were bumped to 0.8.3.
**What was built to handle it:** The agent updated both files in the same bump commit (`35fc684`).
**Should we have known this?** Not from SwiftTuberia's side, no. This is a SwiftAcervo-side hygiene issue — its release tooling does not bump both version locations atomically.
**Carry forward:** File a small follow-up issue on `intrusive-memory/SwiftAcervo` to add a release script (or update `make release`) that bumps both `Sources/SwiftAcervo/Acervo.swift` and `Sources/acervo/Version.swift` in lock-step. Out of scope for this mission, but worth tracking.

### 4. There is an orphan SwiftAcervo commit `c8bb6d8` from earlier today

**What happened:** During brief preparation, the supervisor discovered an unmerged SwiftAcervo commit `c8bb6d8 feat(ComponentHandle): add public rootDirectoryURL accessor` authored by the user at 07:46 today (before this mission ran). It sits on a parallel branch off `83dd8f6`, never merged into `development` or `main`. It adds 13 lines (vs. the mission's 1-line `207be7e`) — likely the same accessor with docstring or additional context.
**What was built to handle it:** Nothing — the mission's `207be7e` is what made it into the v0.8.3 tag. `c8bb6d8` is harmlessly orphaned.
**Should we have known this?** Yes — checking `git log --all` on SwiftAcervo before dispatching Acervo-S1 would have surfaced it. The mission supervisor's startup protocol does not currently inspect parallel branches in dependency repos.
**Carry forward:** When a plan touches a sibling repo, the startup protocol should `git log --all --oneline` that repo and surface any in-progress branches that the user might want to consume from. Also: the user should reconcile `c8bb6d8` (cherry-pick its docstring contents into 0.8.4 if better, or `git branch -D` the orphan branch).

---

## Section 2: Process Discoveries

### What the Agents Did Right

#### 1. First-attempt completion on all 5 sorties

**What happened:** Acervo-S1, Acervo-S2, Tuberia-S1, Tuberia-S2, Tuberia-S3 all completed on attempt 1 of 3. Zero PARTIAL, zero BACKOFF, zero FATAL.
**Right or wrong?** Right. The sergeant prompt structure (one goal, lean context, machine-verifiable exit criteria) plus the iteration-01 carry-forward rules (sonnet-floor on release, U4 auth precondition, U1 commit gate) sized the work correctly and de-risked the failure modes. Cost-effective — 5 sonnet dispatches, no haiku-then-sonnet escalation overhead.
**Evidence:** SUPERVISOR_STATE.md shows attempt counters at 1/3 across the board. Git log on `mission/vanishing-manifest/02` has exactly 4 commits since `36887b9` (the mission setup + 3 Tuberia sorties), one per sortie, U1-compliant subjects.
**Carry forward:** Keep the sergeant prompt template. Keep the universal exit criteria block. Keep sonnet-floor on release/public-artifact sorties.

#### 2. Tuberia-S2 caught a `FileManager` reference in a doc comment

**What happened:** Tuberia-S2's exit criterion was `grep -n 'FileManager' WeightLoader.swift` returns 0. After deleting the two dead methods, one mention of `FileManager` survived in a doc comment ("Tries `FileManager.enumerator` first…"). The agent rewrote the comment to "file system access" — preserving the spirit (no `FileManager`-coupled language) without changing behavior.
**Right or wrong?** Right. Initiative within the spirit of the criterion, no scope creep, transparently reported in the agent's debrief.
**Carry forward:** Exit criteria expressed as "grep returns 0" can be tripped by comments and dead text. Agents should treat this as an opportunity to fix the comment, not as a reason to halt — provided the change doesn't alter behavior.

#### 3. Tuberia-S3 pivoted from the broken `-resolvePackageDependencies` to `swift package update`

**What happened:** See Hard Discovery #1. The agent could have either halted ("plan command doesn't work") or pivoted to a permitted alternative. It pivoted, succeeded, and surfaced the deviation in its report.
**Right or wrong?** Right. The pivot was within CLAUDE.md's allowed set (only `swift build`/`swift test` are forbidden) and the deviation was logged for review.
**Carry forward:** Agents should treat `xcodebuild -resolvePackageDependencies` failures (in version-bump scenarios) as a known plan-command bug and pivot to `swift package update <name>` without retry-loop overhead.

### What the Agents Did Wrong

#### 1. Tuberia-S3 force-added a gitignored file without surfacing the conflict

**What happened:** When Tuberia-S3 noticed `Package.resolved` was gitignored, it used `git add -f` and proceeded. It mentioned this in the post-hoc "surprises" section. The supervisor noticed and asked the user.
**Right or wrong?** Wrong, mildly. Force-adding overrode an explicit project-level convention. The right behavior would have been to halt, surface the conflict ("the plan says commit Package.resolved, but it's gitignored — proceed with `git add -f`, or pivot to a Package.swift-constraint approach?"), and let the supervisor decide. This is a `Package.swift` vs `Package.resolved` policy question, not an implementation decision the agent should resolve unilaterally.
**Evidence:** `git ls-files Package.resolved` returns the file (tracked), `grep Package.resolved .gitignore` also returns it (still ignored on paper).
**Carry forward:** Agent prompts for sorties that touch generated/resolved files should include an explicit instruction: "If the file is in `.gitignore`, HALT and surface to the supervisor before deciding to track it."

### What the Planner Did Wrong

#### 1. Refinement replaced `swift package update` with `xcodebuild -resolvePackageDependencies` in error

**What happened:** The auto-fix table in EXECUTION_PLAN.md explicitly notes: "swift package update is excluded per CLAUDE.md (only swift build and swift test are forbidden, but intent was xcodebuild) → AUTO-FIXED: Changed to xcodebuild -resolvePackageDependencies". This refinement misread CLAUDE.md (which only bans `swift build`/`swift test`, not `swift package update`) and replaced the correct tool with a less-correct one.
**Right or wrong?** Wrong. The refinement was over-cautious about CLAUDE.md's scope and chose a tool that doesn't actually accomplish the bump.
**Evidence:** Tuberia-S3's pivot in the field. Three minutes wasted on the wrong command before the agent figured out the correct one.
**Carry forward:** Refinement passes that exclude `swift X` commands need to read CLAUDE.md *literally*. CLAUDE.md says "NEVER use `swift build` or `swift test`" — that is the entire prohibited set. `swift package update`, `swift package resolve`, `swift package describe`, etc. are all allowed.

#### 2. Plan did not bump `Package.swift`'s `from:` constraint

**What happened:** The plan only updated `Package.resolved` (the concrete pin). Tuberia's `Package.swift` still says `.upToNextMajor(from: "0.7.3")`. A fresh `swift package update` (no version arg) by any consumer could resolve to `0.7.3`, `0.8.0`, `0.8.1`, or `0.8.2` — none of which have `rootDirectoryURL`. The build would fail.
**Right or wrong?** Wrong. The semantic guarantee is "this package requires SwiftAcervo ≥ 0.8.3", and that guarantee belongs in `Package.swift`, not in `Package.resolved`.
**Evidence:** Currently `Package.swift:24` says `from: "0.7.3"` while the actual minimum API requirement is 0.8.3. Latent bug.
**Carry forward:** Any plan that newly consumes a remote-package symbol should include a `from:` bump to the minimum supporting version, in the same sortie that adds the consumption. One-line addition; same commit.

#### 3. The plan did not anticipate `Package.resolved` being gitignored

**What happened:** See Hard Discovery #2. The plan stipulated committing `Package.resolved` without checking that it could be committed.
**Carry forward:** See HD#2 carry-forward (gitignore audit during refinement).

#### 4. The "supervising agent" / "sub-agent" parallelism vocabulary was confusing

**What happened:** The plan's parallelism structure described a single "supervising agent" that handles all build sorties serially, plus a "sub-agent (Agent 2)" for the release sortie in parallel. The mission supervisor's standard model is "every sortie is its own background agent." There was a 30-second moment of interpretation at start time before the supervisor decided to map "supervising agent" → "regular sub-agent dispatched serially within each repo, parallel across repos."
**Right or wrong?** The plan's vocabulary was inherited from iteration 01 mental model. The interpretation worked, but the vocabulary is a stumbling block.
**Carry forward:** Refinement should rewrite plans in mission-supervisor-native terminology: "background agents per sortie, dependency-gated." Drop "supervising agent" / "Agent 2" naming. The supervisor *is* the dispatcher; agents are just agents.

---

## Section 3: Open Decisions

### 1. Track `Package.resolved` or not?

**Why it matters:** Currently force-tracked but still listed in `.gitignore`. The two are inconsistent. Future contributors will be confused about whether to commit changes to it.
**Options:**
- **A.** Keep tracked. Remove `Package.resolved` from `.gitignore` to make intent explicit. (App / leaf-consumer convention.)
- **B.** Untrack via `git rm --cached Package.resolved` (in a follow-up commit), keep gitignored, and lock the lower bound via `Package.swift` constraint instead. (Library convention.)
**Recommendation:** **B.** SwiftTuberia is published as a Swift Package (the `Tuberia` library product). Library convention is to NOT track `Package.resolved` — consumers resolve their own. Bump `Package.swift` to `.upToNextMajor(from: "0.8.3")` and untrack `Package.resolved`. Net effect: same API guarantee, simpler convention, no gitignore drift.

### 2. Bump `Package.swift` constraint to `from: "0.8.3"`?

**Why it matters:** Without the bump, `swift package update` (no args) by any consumer could resolve to 0.7.3 → 0.8.2, none of which have `rootDirectoryURL`. Latent build break.
**Options:**
- **A.** Bump to `.upToNextMajor(from: "0.8.3")`.
- **B.** Leave at `0.7.3` and rely on `Package.resolved`.
**Recommendation:** **A**, regardless of decision #1. The `from:` constraint is the authoritative API guarantee.

### 3. Run Acervo-S3 (optional `estimatedSizeBytes` hint)?

**Why it matters:** Currently no consumer needs `estimatedSizeBytes` on bare descriptors. Running S3 would require a v0.8.4 release on SwiftAcervo (since v0.8.3 is shipped). That's a real cost (CI run, version bump, release notes, repo churn) for zero current benefit.
**Options:**
- **A.** Skip permanently; remove from plan and from `requirements/ACERVO_V2_COMPLIANCE.md`.
- **B.** Defer to a future mission when a consumer actually needs it (file as a tracked GitHub issue on SwiftAcervo).
**Recommendation:** **B.** File as an issue with "blocked-on-consumer-need" label. If a consumer surfaces, run S3 then.

### 4. Mission close-out path

**Why it matters:** The mission branch sits on 4 commits. To "ship" it requires integration into `main`.
**Options:**
- **A.** PR `mission/vanishing-manifest/02 → main`, run CI, merge. Per-sortie commit history preserved.
- **B.** Direct merge (faster but bypasses CI).
- **C.** Squash to a single commit on `main`, lose the audit trail.
**Recommendation:** **A.** The per-sortie commits are valuable (one of the iteration-01 carry-forward wins); CI verification via PR is the right gate.

### 5. Reconcile orphan SwiftAcervo commit `c8bb6d8`

**Why it matters:** It's an unused commit you authored manually before this mission. It will linger on a local branch indefinitely.
**Options:**
- **A.** Cherry-pick its docstring/additions onto `development` for inclusion in v0.8.4 (if it has useful content beyond the bare accessor).
- **B.** `git branch -D` the parallel branch (or let it expire by reflog).
- **C.** Inspect first (`git show c8bb6d8`), then decide.
**Recommendation:** **C** then **B**. Almost certainly safe to discard; verify briefly to make sure no useful content is being lost.

---

## Section 4: Sortie Accuracy

| Sortie | Task | Model | Attempts | Accurate? | Notes |
|--------|------|-------|----------|-----------|-------|
| Acervo-S1 | rootDirectoryURL accessor | sonnet | 1/3 | ✅ Yes | Single-file, 1-line edit. Surgical. Commit `207be7e`. |
| Acervo-S2 | Release v0.8.3 | sonnet | 1/3 | ✅ Yes | Surfaced the CLI version drift (HD#3) and bumped both files. Tag `652856b`, bump `35fc684`, release published 14:39:18Z. |
| Acervo-S3 | estimatedSizeBytes hint | — | — | n/a | Skipped (optional; see Open Decision #3). |
| Tuberia-S1 | Bare ComponentDescriptors + deprecate | sonnet | 1/3 | ✅ Yes | Updated 3 test assertions cleanly to match bare-descriptor semantics. Commit `df285bd`. |
| Tuberia-S2 | Typed AcervoError + dead code removal | sonnet | 1/3 | ✅ Yes | All 4 AcervoError cases mapped (including `integrityCheckFailed`, the iter-01-blind-spot case). Caught `FileManager` in a doc comment and rewrote it (Right-3). Commit `91ad4ed`. |
| Tuberia-S3 | rootDirectoryURL in T5XXLEncoder | sonnet | 1/3 | ⚠️ Mostly | Pivoted from broken `-resolvePackageDependencies` to `swift package update` (Right-3). Force-added gitignored `Package.resolved` without halting (Wrong-1). Build/test pass; deviations defensible but the gitignore one was a halt-and-ask, not a unilateral call. Commit `ee0126c`. |

**Aggregate:** 5/5 sorties produced commits that survive into the final state with no rework. One sortie (Tuberia-S3) made two field-decisions, both defensible, one of which should have been escalated.

---

## Section 5: Harvest Summary

The iteration-01 carry-forward rules paid for themselves on first contact. Per-sortie commit gates produced an audit trail that made verification trivial. Sonnet-floor on the release sortie meant Acervo-S2 caught the CLI version drift cleanly instead of hand-waving past it. The U4 auth precondition cost nothing (gh auth was live throughout) but would have prevented a repeat of iter-01's silent-abandonment failure mode if it hadn't been. Sergeant-style prompts (one goal, lean context, machine-verifiable exit criteria) put all five sorties on first-attempt success — five sonnet dispatches, zero retries, zero PARTIAL.

The single most important new lesson: **`xcodebuild -resolvePackageDependencies` does not upgrade pinned SPM versions.** Plans that need a SPM bump should call `swift package update <name>` explicitly. CLAUDE.md's `swift build`/`swift test` ban does NOT extend to `swift package update`. Refinement passes need to read CLAUDE.md literally, not generously.

The second-most-important lesson: **planner refinement must check `.gitignore` before stipulating that a file is "touched."** A file that's gitignored requires a deliberate planner decision (track it, untrack it, or use a different mechanism) — not an agent-level field decision.

What changes for the next iteration: the next plan's refinement adds a `.gitignore` audit step in Pass 4 (open questions), and the next plan's commands prefer `swift package update <name>` over `xcodebuild -resolvePackageDependencies` when the goal is a version bump.

---

## Section 6: Files

### Preserve (read-only reference)

| File | Branch | Why |
|------|--------|-----|
| `mission/vanishing-manifest/02` (commits `f879e06`, `df285bd`, `91ad4ed`, `ee0126c`) | local | Mission audit trail, per-sortie commits. Will become PR target. |
| SwiftAcervo `development` (commits `207be7e`, `35fc684`) | `intrusive-memory/SwiftAcervo:development` | Acervo-S1/S2 commits already pushed. v0.8.3 tag published. |
| `requirements/ACERVO_V2_COMPLIANCE.md` (committed in `f879e06`) | `mission/vanishing-manifest/02` | Source-of-truth requirements doc. |
| Iteration 01 briefs in `docs/complete/` (`vanishing-manifest-01-brief.md`, `vanishing-manifest-02-brief.md`) | already on main | Lessons that drove iteration 02's success. |

### Discard

| File | Why it's safe to lose |
|------|-----------------------|
| `EXECUTION_PLAN.md` (in workspace, after archive) | Mission complete; plan served its purpose. Iteration metadata captured in this brief. |
| `SUPERVISOR_STATE.md` | Per skill spec, removed at archive time. Final state captured here. |

**Note on rollback:** This mission is **Complete**, not "discard and iterate." The rollback ritual should NOT fire. The mission branch should be PR'd to `main`, not reset.

---

## Iteration Metadata

**Starting point commit:** `36887b9` (`deps: migrate tokenizer package — swift-transformers → swift-tokenizers`)
**Mission branch:** `mission/vanishing-manifest/02`
**Final commit on mission branch:** `ee0126c` (`Tuberia-S3: T5XXLEncoder uses rootDirectoryURL + precondition docstring`)
**SwiftAcervo final commit:** `35fc684` on `main`, tag `v0.8.3` (`652856b`)
**Rollback target:** **NOT APPLICABLE.** Mission complete; recommended path is PR to `main`, not rollback.
**Next iteration branch:** **NOT APPLICABLE** unless DoD changes or follow-up scope (Open Decisions 1, 2, 5) is bundled into a small follow-up mission. Decisions 1+2 are 2-line changes — likely better as direct commits to `mission/vanishing-manifest/02` (or to `main` post-merge) rather than a new iteration.
