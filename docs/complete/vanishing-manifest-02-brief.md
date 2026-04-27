# Iteration 01 Brief (Final) — OPERATION VANISHING MANIFEST

> **Terminology:** *Mission* = the definable scope of work. *Sortie* = one atomic agent dispatch within that mission. *Brief* = post-mission review.

> **Numbering note:** The EXECUTION_PLAN.md frontmatter still says `iteration: 1`. An interim brief was written and archived as `docs/complete/vanishing-manifest-01-brief.md` while the mission was paused on a `gh` auth blocker. This is the **second pass** on iteration 01 — written after the user declared the mission over without resuming. Filename uses `02` to avoid colliding with the archived interim brief; the iteration in plan frontmatter is unchanged.

**Mission:** Make SwiftTuberia never deal with files — Acervo provides all model access (V2-compliance requirements across SwiftAcervo + SwiftTuberia).
**Branch:** `mission/vanishing-manifest/01`
**Starting Point Commit:** `36887b9` (deps: migrate tokenizer package — swift-transformers → swift-tokenizers)
**Sorties Planned:** 6 (5 required + 1 optional)
**Sorties Completed:** 3 (Acervo-S1, Tuberia-S1, Tuberia-S2)
**Sorties Failed/Blocked:** 0 FATAL; 1 PARTIAL (Acervo-S2, never resumed); 1 PENDING (Tuberia-S3, never started); 1 OPTIONAL-skipped (Acervo-S3)
**Duration:** ~1 working session, 2026-04-27. Cost-relative: 3 sonnet sorties + 1 haiku sortie. Cheap.
**Outcome:** **Incomplete (Abandoned)**
**Verdict:** **The mission did not meet its Definition of Done. Three of the six sorties never crossed their finish line, and the Definition of Done's clause 3 (`grep -rn 'deletingLastPathComponent' Sources/TuberiaCatalog/` returns 0) still fails. The committed work is real and valuable — keep it on `mission/vanishing-manifest/01` — but don't pretend the mission is "complete." It is being closed early. The unfinished work is small (≤ 2 sorties of effort, all blockers cleared) and should be picked up in a follow-up mission, not silently dropped.**

---

## Section 1: Hard Discoveries

The interim brief (`docs/complete/vanishing-manifest-01-brief.md`) covered the three hard discoveries from this mission in detail:
1. **`gh` auth tokens expire silently mid-mission** — caused the original Acervo-S2 PARTIAL.
2. **Tuberia's old error path was already broken, not just ugly** — the typed-error rewrite caught a real gap (integrity-failed mapping had no string match at all).
3. **Bare `ComponentDescriptor` initializer already existed in SwiftAcervo** — confirmed Acervo-S3 is correctly scoped as optional.

No new hard discoveries surfaced after the interim brief, because **no new agent work was done.** The only material change in the world since the interim brief is:

### 4. Restoring `gh` auth alone does not resume a mission

**What happened:** The interim brief identified the blocker (expired `gh` token) and prescribed `/mission-supervisor resume` after re-auth. The auth was restored (`gh auth status` now reports `Logged in to github.com account stovak`), but `resume` was never invoked. The mission sat idle until the user declared it over.
**What was built to handle it:** Nothing.
**Should we have known this?** Yes. A brief that ends with "resume after the human does X" creates a checkpoint that decays if X happens silently and nobody looks at the brief afterward. The supervisor has no daemon watching for `gh` auth restoration, and the human has no automatic reminder.
**Carry forward:** When a mission pauses on a human-action blocker (auth, network, manual approval), the brief should include an explicit "trigger condition" that, when satisfied, the supervisor checks for on the *next* invocation. E.g., on next `/mission-supervisor status`, if `gh auth status` exits 0 and the prior brief flagged it as the blocker, surface a "READY TO RESUME" banner. Without that, paused missions become silently abandoned missions.

---

## Section 2: Process Discoveries

### What the Agents Did Right

#### 1. Recommendation C from the interim brief was followed cleanly

**What happened:** The interim brief recommended committing the staged Tuberia-S1 and Tuberia-S2 work as separate commits before resuming. This was done: `7d62e61` (Tuberia-S1) and `6dc20a1` (Tuberia-S2) are now on the branch with focused, accurate commit messages.
**Right or wrong?** Right. The repo now has a per-sortie audit trail. If a future iteration needs to bisect or revert a specific sortie's contribution, it can.
**Evidence:** `git log 36887b9..HEAD` shows 4 commits with one-sortie-per-commit granularity for the two completed Tuberia sorties (plus the deps bump and the mission-docs commit).
**Carry forward:** The "per-sortie commit" rule the interim brief argued for as a hard rule has been validated by retrofit. Bake it into the planner: every code sortie's exit criteria must include `git log -1 --format=%s | grep -F '<sortie-id>'` (or equivalent) before COMPLETED.

### What the Agents Did Wrong

#### 1. The mission was abandoned by inaction, not decision

**What happened:** From the interim brief's writing through this final brief, no sortie was dispatched. No FATAL, no STOPPED, no STOPPING — just silence on `BLOCKED → never resumed`. The state machine has no terminal "Abandoned" state; the de facto path was: mission goes idle, user works on something else, eventually says "I think the mission is over."
**Right or wrong?** Wrong. Missions should end with a deliberate decision (Complete / Abandoned-with-reason / Rolled-back), not by attrition.
**Evidence:** Six-day gap between the interim brief commit and this final brief, with zero sortie activity in between. SUPERVISOR_STATE.md was never updated to reflect the cleared `gh` auth blocker.
**Carry forward:** Add an "ABANDONED" terminal state to the state machine. Require a brief to enter it, with a stated reason. Make `/mission-supervisor status` warn when a mission has been BLOCKED for > N invocations without progress, surfacing the question "do you intend to resume or abandon?" rather than letting it drift.

### What the Planner Did Wrong

The interim brief's three planner critiques (release sortie had no auth precondition; haiku for release sortie was borderline; no "all sorties have committed work" gate) all stand and are not repeated here. One additional finding:

#### 1. The plan had no exit-or-abandon decision point

**What happened:** EXECUTION_PLAN.md ended with a Definition of Done but no guidance on what to do if the DoD partially holds. There was no defined criterion for "this is enough — stop here and merge what we have," nor for "this much shortfall warrants a rollback." When the user lost momentum, the plan provided no way to land the mission gracefully.
**Right or wrong?** Wrong. Real engineering missions get descoped. Plans should anticipate this.
**Evidence:** The current state — three sorties committed, two undone, DoD failing — is not represented anywhere in the plan as a possible terminal state. There's no answer to "is this state shippable?"
**Carry forward:** Add a "Minimum Viable Outcome" section to plan templates, distinct from Definition of Done. MVO defines the smallest state worth landing on main; DoD defines full success. A mission can be abandoned successfully if MVO is met.

---

## Section 3: Open Decisions

### 1. What happens to the committed work on `mission/vanishing-manifest/01`?

**Why it matters:** The branch holds three good commits: a deps bump, two sortie commits with passing tests, and the mission-docs commit. If nothing is done, this work rots on a branch that will never merge. If the wrong thing is done, partial V2-compliance state lands on main and confuses future readers.

**Options:**
- **A. Merge to main as-is.** Pro: ~150 lines of real improvement (typed AcervoError handling, bare descriptors, deprecated `ensureComponentReady`) ship. Con: T5XXLEncoder still uses `deletingLastPathComponent()`, so V2 compliance is *partial* on main without that being obvious to a future reader. The deprecated `ensureComponentReady` may also have callers downstream that haven't been migrated.
- **B. Merge to main with a known-issues note.** Same as A, plus a short note in `requirements/ACERVO_V2_COMPLIANCE.md` (or AGENTS.md) saying "T5XXLEncoder.loadTokenizer() still uses path arithmetic; awaiting SwiftAcervo v0.8.3 with `rootDirectoryURL`." Pro: future-reader clarity. Con: the note becomes stale if the V2 work resumes.
- **C. Leave the branch unmerged; open a follow-up mission.** Pro: nothing partial lands on main. Con: branch decay; the work is invisible to anyone not looking at the branch list.
- **D. Roll back entirely.** Discard all the committed work. Pro: clean slate. Con: throws away tested, correct improvements that are independent of the V2 endgame (typed errors, bare descriptors). **Bad trade.**

**Recommendation:** **B.** The typed-error refactor and bare-descriptor work are good on their own merits and shouldn't be held hostage to the unfinished S3 work. Add the known-issue note so a future engineer sees the gap.

### 2. Does SwiftAcervo's local `c8bb6d8` get pushed, or rolled back?

**Why it matters:** `../SwiftAcervo` has a local commit on `development` that adds `public var rootDirectoryURL: URL { baseDirectory }`. It's correct, tested, and unpushed. Three options:

**Options:**
- **A. Push it as a regular development commit; tag/release v0.8.3 later when needed.** Cheapest path. Pro: the commit is preserved upstream; another consumer might benefit. Con: dangling unreleased feature on `development`.
- **B. Push it AND complete the release (tag v0.8.3, gh release).** Pro: closes Acervo-S2 properly; SwiftTuberia-S3 becomes unblocked for a future mission. Con: publishing a release creates the implicit promise that someone is consuming it; if no one is, it's noise.
- **C. Roll it back (`git reset --hard 83dd8f6` on Acervo's `development`).** Pro: clean. Con: discards a real, useful API addition.

**Recommendation:** **A** if the V2 work is truly being abandoned. **B** if it's just paused and might resume. **Not C.**

### 3. Is the V2 compliance work being abandoned, or just paused?

**Why it matters:** This is the question the user implicitly answered with "I think the mission is over." But the answer determines whether the outstanding work is tracked as a follow-up issue or quietly forgotten.

**Options:**
- **A. Truly abandoned.** Update `requirements/ACERVO_V2_COMPLIANCE.md` to mark TUBERIA-V2-04 (and the relevant Acervo requirements) as descoped, with rationale.
- **B. Paused indefinitely.** File a GitHub issue describing the remaining work (Acervo-S2 release + Tuberia-S3) so it's visible.
- **C. Will resume soon.** Don't archive this brief; leave SUPERVISOR_STATE.md so the next `/mission-supervisor resume` works.

**Recommendation:** **B.** The remaining work is small and well-specified — losing the specification by leaving it only in a brief is worse than spending five minutes on an issue.

---

## Section 4: Sortie Accuracy

| Sortie | Task | Model | Attempts | Accurate? | Notes |
|--------|------|-------|----------|-----------|-------|
| Acervo-S1 | Add `rootDirectoryURL` to ComponentHandle | sonnet | 1/3 | YES | Commit `c8bb6d8` (local on Acervo `development`), 13 lines, single file. Exit criteria all met. **Unpushed.** |
| Acervo-S2 | Release v0.8.3 | haiku | 1/3 (PARTIAL) | NO | Stalled on expired `gh` auth. Blocker later cleared but sortie never resumed. Net effect: never executed. Treat as wasted dispatch. |
| Acervo-S3 | `estimatedSizeBytes` hint (optional) | — | 0 | N/A | Optional, never dispatched. Acceptable skip. |
| Tuberia-S1 | Bare descriptors + deprecation | sonnet | 1/3 | YES | Commit `7d62e61`. Tests green. Exit criteria all met. |
| Tuberia-S2 | Dead-code removal + typed errors | sonnet | 1/3 | YES | Commit `6dc20a1`. 29/29 tests green. Caught the integrity-failed mapping gap (see interim brief). |
| Tuberia-S3 | rootDirectoryURL + docstring | — | 0 | NO | Never dispatched (blocked on Acervo-S2 release that never happened). DoD clause 3 still fails because of this. |

**Summary:** 3/6 sorties produced correct work. 1 PARTIAL was abandoned mid-flight. 1 was blocked by the PARTIAL and never started. 1 was optional and skipped. **Accuracy of dispatched-and-attempted work is high; coverage of the planned mission is 50%.**

---

## Section 5: Harvest Summary

The mission produced three correct, tested, committed sorties whose value is independent of the unfinished endgame — those should ship. It also produced two unfinished sorties (`Acervo-S2`, `Tuberia-S3`) that, taken together, are the actual point of the V2-compliance mission: removing path arithmetic from `T5XXLEncoder.loadTokenizer()`. Without them, the mission's headline goal isn't achieved. The single most important lesson is operational, not technical: **a mission paused on a human-action blocker becomes a silently abandoned mission unless the supervisor has an explicit re-check on the blocker condition.** The interim brief correctly diagnosed the auth issue and gave a clean recovery path; the recovery path was never taken because nothing surfaced the cleared blocker. The fix is a "trigger condition" mechanic: brief states the condition that means "ready to resume," and `/mission-supervisor status` checks it on each invocation.

The technical lesson, restated for the carry-forward pile: every code sortie produces exactly one commit (now validated by the post-hoc commit retrofit), and any sortie that touches credentials or external publishing needs an entry-criterion verifying the credential surface before doing any work.

---

## Section 6: Files

### Preserve (read-only reference for next iteration / follow-up)

| File | Branch | Why |
|------|--------|-----|
| `OPERATION_VANISHING_MANIFEST_02_BRIEF.md` (this file) | `mission/vanishing-manifest/01` (will be archived to `docs/complete/`) | Final brief — captures the abandoned-by-inaction outcome and the three open decisions |
| `docs/complete/vanishing-manifest-01-brief.md` | `mission/vanishing-manifest/01` | Interim brief — preserves the mid-mission diagnosis (auth blocker, missing per-sortie commits) for historical context |
| `EXECUTION_PLAN.md` | `mission/vanishing-manifest/01` | The 6-sortie plan — useful as the spec for any follow-up mission picking up Acervo-S2 + Tuberia-S3 |
| `requirements/ACERVO_V2_COMPLIANCE.md` | `mission/vanishing-manifest/01` | Source-of-truth requirements — should be updated to reflect descoped/deferred items per Open Decision #3 |
| `Sources/Tuberia/Infrastructure/WeightLoader.swift` (committed in `6dc20a1`) | `mission/vanishing-manifest/01` | Tuberia-S2 work — typed AcervoError pattern matching. Independent value; should land on main per Open Decision #1.B |
| `Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` (committed in `7d62e61`) | `mission/vanishing-manifest/01` | Tuberia-S1 work — bare descriptors + `@available` deprecation. Independent value; should land on main per Open Decision #1.B |
| `Tests/TuberiaCatalogTests/CatalogRegistrationTests.swift` (committed in `7d62e61`) | `mission/vanishing-manifest/01` | Test updates aligned with bare descriptors |
| `../SwiftAcervo` commit `c8bb6d8` (local, `development`) | `development` | Acervo-S1 work — needs decision per Open Decision #2 |

### Discard (will not exist after rollback)

| File | Why it's safe to lose |
|------|----------------------|
| (none recommended) | **No rollback is recommended.** The committed work is independently valuable and should be preserved (merged or kept on the branch). If the user overrides and rolls back anyway, the staged/committed sortie work would still survive on `mission/vanishing-manifest/01` for reference. |

---

## Section 7: Iteration Metadata

**Starting point commit:** `36887b9` (deps: migrate tokenizer package — swift-transformers → swift-tokenizers)
**Mission branch:** `mission/vanishing-manifest/01`
**Final commit on mission branch:** `4da4d9b` (mission: OPERATION VANISHING MANIFEST plan, state, brief, requirements)
**Rollback target:** `36887b9` *(only if Open Decision #1.D is taken — NOT recommended)*
**Next iteration branch:** *(none — mission is being abandoned, not iterated)* — if the V2 endgame work is later picked up, prefer a fresh, narrowly-scoped mission branch (e.g., `mission/v2-endgame/01`) rather than `mission/vanishing-manifest/02`, since the surviving work has been merged.

---

## Recommended Next Action

The mission is being closed without meeting its Definition of Done. To close it cleanly:

1. **Resolve Open Decision #1** — recommend **B** (merge `mission/vanishing-manifest/01` to main with a known-issue note about T5XXLEncoder still using `deletingLastPathComponent()`).
2. **Resolve Open Decision #2** — recommend **A** (push `c8bb6d8` to `intrusive-memory/SwiftAcervo` `development`; defer the v0.8.3 tag/release until someone consumes `rootDirectoryURL`). If V2 work is paused rather than abandoned, do **B** instead (full release).
3. **Resolve Open Decision #3** — recommend **B** (file a follow-up issue capturing the remaining Acervo-S2 release + Tuberia-S3 work).
4. Archive this brief to `docs/complete/vanishing-manifest-02-brief.md` and clean up `SUPERVISOR_STATE.md`.

The `mission/vanishing-manifest/01` branch can be preserved locally indefinitely — it's small. Don't delete it.
