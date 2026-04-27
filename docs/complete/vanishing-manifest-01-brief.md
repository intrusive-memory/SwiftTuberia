# Iteration 01 Brief — OPERATION VANISHING MANIFEST

> **Terminology:** *Mission* = the definable scope of work. *Sortie* = one atomic agent dispatch within that mission. *Brief* = post-mission review.

**Mission:** Make SwiftTuberia never deal with files — Acervo provides all model access (8 V2-compliance requirements across SwiftAcervo + SwiftTuberia).
**Branch:** `mission/vanishing-manifest/01`
**Starting Point Commit:** `36887b9` (deps: migrate tokenizer package — swift-transformers → swift-tokenizers)
**Sorties Planned:** 6 (5 required + 1 optional)
**Sorties Completed:** 3 (Acervo-S1, Tuberia-S1, Tuberia-S2) + 1 PARTIAL (Acervo-S2: local commit only)
**Sorties Failed/Blocked:** 0 FATAL; 1 PARTIAL (Acervo-S2); 1 PENDING-blocked (Tuberia-S3); 1 OPTIONAL-skipped (Acervo-S3)
**Duration:** Single session, 2026-04-27. Cost-relative: 3 sonnet sorties + 1 haiku sortie. Cheap.
**Outcome:** Incomplete
**Verdict:** **Partial salvage. Do not roll back.** The completed work is correct, well-scoped, and commit-worthy. The mission is paused on a purely environmental blocker (expired `gh` auth token). Restore auth, resume — do not discard.

---

## Section 1: Hard Discoveries

### 1. `gh` auth tokens expire silently mid-mission

**What happened:** SwiftAcervo-S2 (the release sortie) successfully bumped the version, committed `c8bb6d8` locally, then stalled on `git push` / `gh release create` because the GitHub CLI auth token had expired. The sub-agent reported PARTIAL and held. No FATAL — the supervisor correctly identified this as a non-code blocker rather than retrying.
**What was built to handle it:** Nothing. The sortie was paused with state PARTIAL, awaiting user re-auth. No code workaround attempted (correct call — auth is the user's domain).
**Should we have known this?** Yes. Any sortie that touches `gh` or `git push` to a remote depends on credential state that the planner cannot observe at plan time. We did not add `gh auth status` as an entry criterion.
**Carry forward:** Every release sortie must include an entry criterion: `gh auth status` exits 0 AND `git -C <repo> push --dry-run origin HEAD` exits 0. If either fails, halt before doing any work and prompt the user. Today the partial commit is harmless, but a more invasive release sortie that did the version bump first would have left the repo dirty.

### 2. Tuberia's old error path was already broken — not just ugly

**What happened:** WeightLoader's old catch block matched on stringified error descriptions (`errorString.contains("NotDownloaded")`). Tuberia-S2 replaced this with typed `AcervoError` pattern matching across `componentNotDownloaded`, `componentNotRegistered`, `componentNotHydrated`, `integrityCheckFailed`. The old string-match would have silently failed if Acervo ever changed an error's `String(describing:)` form, and would have missed `integrityCheckFailed` entirely (no string match for it).
**What was built to handle it:** Two parallel switch blocks (in `load()` and `loadFromPath()`) that pattern-match on `AcervoError` cases and translate to `PipelineError`. 8 `AcervoError.` references in the resulting file (exit criterion required ≥ 3).
**Should we have known this?** Partially. The audit (`AUDIT_FINDINGS.md`, now deleted) flagged the string inspection. What we did not anticipate: the integrity-check case had no string match at all, so integrity failures were being mapped to a generic "weightLoadingFailed" with the raw description rather than a typed `PipelineError`. The sortie quietly fixed this gap.
**Carry forward:** When auditing string-based error handling for replacement, enumerate the typed-error cases first and map each one. Don't trust that the old code "covered" the cases it appeared to — it may only have covered the ones whose `String(describing:)` happened to contain the search keyword.

### 3. Bare `ComponentDescriptor` initializer already existed in SwiftAcervo

**What happened:** Tuberia-S1 needed to swap `ComponentDescriptor(... files:, estimatedSizeBytes:, ...)` for the bare form. The plan assumed the bare initializer existed. It did. No initializer addition was required on the Acervo side for this sortie. The plan correctly scoped Acervo-S3 (estimatedSizeBytes hint) as **optional** because it isn't required to make the bare form work.
**What was built to handle it:** Nothing extra — Tuberia-S1 just used the existing initializer.
**Should we have known this?** Yes, and we did. The plan's structure (S3 optional, no Acervo prerequisite for Tuberia-S1) was correct. Calling this out so it doesn't get re-litigated next iteration.
**Carry forward:** None — this is a confirmation, not a constraint. If Acervo-S3 is added in iteration 2 it should remain optional.

---

## Section 2: Process Discoveries

### What the Agents Did Right

#### 1. Tight, single-purpose sorties completed first-try with sonnet

**What happened:** Acervo-S1, Tuberia-S1, Tuberia-S2 each completed in 1/3 attempts with sonnet. Exit criteria were grep-checkable and all passed. 29/29 tests green in Tuberia after S2.
**Right or wrong?** Right. Sortie sizing matched complexity. No retries, no PARTIAL, no rework.
**Evidence:** Three SUPERVISOR_STATE.md "Completed Sorties" rows, each at attempt 1/3. Diffs are surgical: Acervo-S1 = 13 lines added to one file; Tuberia-S1 = ~40-line deletion of dead arrays + bare-initializer swap + one `@available`; Tuberia-S2 = ~45 lines deleted (`canEnumerateDirectory`, `findSafetensorsFiles`) + two typed catch blocks.
**Carry forward:** Keep this sizing target. Atomic file-or-symbol-scoped sorties with grep-verifiable exit criteria converge in one shot. Don't widen.

#### 2. Sub-agent boundary was respected

**What happened:** Per the plan, only Acervo-S2 (release, no build) ran on the sub-agent. All build sorties stayed on the supervising agent. No xcodebuild conflicts.
**Right or wrong?** Right. The constraint was enforced.
**Evidence:** No build interleaving. The PARTIAL on Acervo-S2 was an auth issue, not a contention issue.
**Carry forward:** Keep "no build on sub-agents." It worked.

### What the Agents Did Wrong

#### 1. No per-sortie commits on the SwiftTuberia mission branch

**What happened:** `git log 36887b9..HEAD` on `mission/vanishing-manifest/01` is **empty**. All Tuberia-S1 and Tuberia-S2 work is staged in the index but never committed. The diff shows 1016 lines of changes across 8 files all sitting in one staged blob.
**Right or wrong?** Wrong. We have no per-sortie audit trail in git history. If we needed to bisect or revert one sortie, we couldn't — they're indistinguishable in the index.
**Evidence:** `git status` shows 6 modified/added files staged plus 1 deleted unstaged (`AUDIT_FINDINGS.md`), with zero commits on the branch since `36887b9`.
**Carry forward:** Each completed sortie must produce a commit before the next sortie dispatches. Sortie-N commit message: `<sortie-id>: <one-line summary>`. The supervisor should verify `git log -1 --format=%s` matches the expected sortie before declaring COMPLETED. This becomes an exit-criterion of every code sortie.

#### 2. `AUDIT_FINDINGS.md` deleted in working tree, not staged

**What happened:** `AUDIT_FINDINGS.md` is shown as `D` (deleted, unstaged). Either an agent removed it expecting the deletion to be staged, or this is leftover from prior work. It's not part of any sortie's exit criteria.
**Right or wrong?** Untracked side-effect. Minor, but exactly the kind of stray edit that pollutes a mission branch.
**Evidence:** `git status` line: `deleted: AUDIT_FINDINGS.md` not staged.
**Carry forward:** Sortie agents should leave the working tree clean except for files their exit criteria name. Verify with `git status --porcelain` after each sortie — only expected paths should appear.

### What the Planner Did Wrong

#### 1. Release sortie had no auth precondition

**What happened:** Acervo-S2's entry criteria checked git tree cleanliness but not credentials. The very first thing the sortie needed (`gh release`, `git push`) requires auth that wasn't verified.
**Right or wrong?** Wrong. The planner enumerated file-system preconditions but missed credential preconditions.
**Evidence:** `gh release view v0.8.3` exit-criterion was checkable, but the path to satisfying it was never gated on `gh auth status`. Sortie did real work (version bump, local commit `c8bb6d8`) before discovering it couldn't push.
**Carry forward:** Any sortie that uses `gh`, `git push`, network HTTP to a credentialed endpoint, or third-party CLI auth must have an entry criterion verifying that auth is live. Better still: the supervisor's `start` and `resume` commands should pre-flight `gh auth status` once before dispatching any release-class sortie.

#### 2. Haiku for the release sortie was a marginal call

**What happened:** Acervo-S2 was assigned haiku (complexity 5). It involves credentialed external calls and is irreversible (creates a public release). When auth failed, the sortie correctly halted, but if it had succeeded with stale instructions it could have published a wrong release.
**Right or wrong?** Defensible but borderline. The work itself is mechanical, justifying haiku on cost. The blast radius (public release tag) argues for sonnet.
**Evidence:** Decisions Log shows Acervo-S1 was upgraded to sonnet specifically because it "establishes public API surface" — Acervo-S2 *publishes* that surface and was downgraded. The cost asymmetry is the wrong direction.
**Carry forward:** Treat any sortie that creates a public artifact (git tag pushed to origin, GitHub release, npm publish, App Store submit) as sonnet-minimum regardless of code complexity. The marginal cost (10x vs 1x for one sortie) is dwarfed by the cost of a bad public release.

#### 3. No "all sorties have committed work" gate before declaring COMPLETED

**What happened:** Tuberia-S1 and Tuberia-S2 were marked COMPLETED in SUPERVISOR_STATE.md based on test results and grep counts. Neither check noticed the work was uncommitted.
**Right or wrong?** Wrong, in the same vein as #1 above. The completion verification was code-correctness only, not repo-hygiene.
**Evidence:** SUPERVISOR_STATE.md "Completed Sorties" table lists Tuberia-S1 and S2 as verified, but `git log 36887b9..HEAD` is empty.
**Carry forward:** Add a universal exit criterion to every code sortie: `git rev-list HEAD --not 36887b9 -- <files-touched>` returns ≥ 1 commit AND that commit's message matches the sortie ID. This is the per-sortie commit gate from agent discovery #1, enforced from the planner side.

---

## Section 3: Open Decisions

### 1. Restore `gh` auth and resume, or roll back?

**Why it matters:** Three sorties of clean, tested work are sitting staged. Rolling back discards them. Resuming requires the user to run `gh auth login` (interactive) and `git -C ../SwiftAcervo push origin development && git tag v0.8.3 && git push origin v0.8.3 && gh release create v0.8.3 --repo intrusive-memory/SwiftAcervo`.
**Options:**
- A. **Resume.** User re-auths, the supervisor continues Acervo-S2 from where it stopped, then dispatches Tuberia-S3. ~2 sorties of work remain.
- B. **Roll back to `36887b9`.** Discard all staged work and restart with the entry-criteria fixes from this brief baked in.
- C. **Commit what we have, branch, then re-auth.** Get the staged work into per-sortie commits *first* (so we have an audit trail), then proceed with A.
**Recommendation:** **C.** The work is good. Lose neither the work nor the audit trail. Commit Tuberia-S1 and Tuberia-S2 as separate commits, then resume Acervo-S2.

### 2. Should `gh auth status` become a universal supervisor pre-flight?

**Why it matters:** This will recur in any mission that releases a library, opens a PR, or pushes a branch. Adding a one-time pre-flight at `start` time is cheap.
**Options:**
- A. Add to per-sortie entry criteria as needed (status quo + rule).
- B. Add to `mission-supervisor start` and `mission-supervisor resume` as a global pre-flight when any sortie in the plan uses `gh` or `git push`.
**Recommendation:** **B.** Pre-flight at start. Surfaces credential issues before any work happens.

### 3. Is per-sortie commit enforcement an opinion or a hard rule?

**Why it matters:** This brief argues for it as a universal rule. It costs little. The objection is that some sorties produce intermediate state that's not commit-worthy in isolation. Counter-argument: if a sortie's output isn't commit-worthy, the sortie was sized wrong.
**Options:**
- A. Hard rule: every code sortie produces exactly one commit; supervisor verifies before COMPLETED.
- B. Soft rule: encouraged, not enforced.
**Recommendation:** **A.** Hard rule. Atomicity in plans should mirror atomicity in git history. If you can't commit it on its own, it isn't atomic.

---

## Section 4: Sortie Accuracy

| Sortie | Task | Model | Attempts | Accurate? | Notes |
|--------|------|-------|----------|-----------|-------|
| Acervo-S1 | Add `rootDirectoryURL` to ComponentHandle | sonnet | 1/3 | YES | One commit `c8bb6d8`, 13 lines, single file. Exemplary. |
| Acervo-S2 | Release v0.8.3 | haiku | 1/3 (PARTIAL) | PARTIAL | Local commit `c8bb6d8` succeeded; push/tag/release blocked on expired `gh` auth. Not the agent's fault — credential issue. |
| Acervo-S3 | `estimatedSizeBytes` hint (optional) | — | 0 | N/A | Optional, not dispatched. Skip is acceptable. |
| Tuberia-S1 | Bare descriptors + deprecation | sonnet | 1/3 | YES (uncommitted) | Code is correct; exit criteria all met; tests pass. **But uncommitted on the branch.** Counts as accurate work, sloppy hygiene. |
| Tuberia-S2 | Dead-code removal + typed errors | sonnet | 1/3 | YES (uncommitted) | Same as S1. 29/29 tests green. Typed `AcervoError` pattern-match is cleaner than the old string check, and caught a real gap (integrity-failed mapping). Uncommitted. |
| Tuberia-S3 | rootDirectoryURL + docstring | — | 0 | N/A | Blocked on Acervo-S2 release. |

**Summary:** 3 fully-accurate sorties, 1 environment-blocked sortie, 0 wasted sorties. The work product is strong; the missing piece is per-sortie commits.

---

## Section 5: Harvest Summary

We learned that this mission's planning was sound at the code level and weak at the operations level. The four sorties that ran each produced correct, tested code on the first attempt — sortie sizing, exit criteria, and model selection were almost all calibrated correctly. What broke us was the unglamorous infrastructure layer: credentials and commits. The supervisor declared sorties complete based on code-correctness checks (greps, tests) but never verified that the work was committed, and the planner never verified that the credentials needed to publish the upstream release were live before the release sortie ran. The single most important change for iteration 2: **make per-sortie commits and live-credential checks first-class exit/entry criteria, not afterthoughts.** Treat the git history and the auth surface as part of the contract, not as ambient infrastructure.

---

## Section 6: Files

### Preserve (read-only reference for next iteration)

| File | Branch | Why |
|------|--------|-----|
| `OPERATION_VANISHING_MANIFEST_01_BRIEF.md` | mission/vanishing-manifest/01 (will be archived to docs/complete) | This brief — carry forward to whatever branch resumes the work |
| `EXECUTION_PLAN.md` (current staged version) | mission/vanishing-manifest/01 | The refined 6-sortie plan; still valid for resume |
| `SUPERVISOR_STATE.md` | mission/vanishing-manifest/01 | Records exact sortie state (Acervo-S2 PARTIAL with commit `c8bb6d8`) — needed to resume from the right point |
| `requirements/ACERVO_V2_COMPLIANCE.md` | mission/vanishing-manifest/01 | Source-of-truth requirements doc; still relevant |
| `Sources/Tuberia/Infrastructure/WeightLoader.swift` (staged) | mission/vanishing-manifest/01 | Tuberia-S2 work — typed AcervoError pattern matching. Commit before resume. |
| `Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` (staged) | mission/vanishing-manifest/01 | Tuberia-S1 work — bare descriptors + @available. Commit before resume. |
| `Tests/TuberiaCatalogTests/CatalogRegistrationTests.swift` (staged) | mission/vanishing-manifest/01 | Test updates aligned with bare descriptors |
| `../SwiftAcervo` commit `c8bb6d8` | development | Acervo-S1 work — needs push + tag v0.8.3 + GitHub release |

### Discard (will not exist after rollback)

| File | Why it's safe to lose |
|------|----------------------|
| (none) | **No rollback recommended.** The mission is incomplete but salvageable; nothing should be discarded. |

If the user overrides and rolls back anyway, the staged work (Tuberia-S1, Tuberia-S2 file edits) would be lost from the working tree but preserved on `mission/vanishing-manifest/01` as a reference branch.

---

## Iteration Metadata

**Starting point commit:** `36887b9` (deps: migrate tokenizer package — swift-transformers → swift-tokenizers)
**Mission branch:** `mission/vanishing-manifest/01`
**Final commit on mission branch:** `36887b9` *(no commits made — all work staged but uncommitted; see process discovery #1)*
**Rollback target:** `36887b9` *(only relevant if rollback is chosen, which is NOT recommended)*
**Next iteration branch:** `mission/vanishing-manifest/02` *(only created if user opts to roll back; default path is to commit + resume on this branch)*

---

## Recommended Next Action

**Do not roll back.** The mission is paused, not failed. Recommended sequence:

1. User runs `gh auth login` to restore GitHub CLI auth.
2. Commit the currently-staged work as two atomic commits on `mission/vanishing-manifest/01`:
   - One commit for Tuberia-S1 (CatalogRegistration + the test file changes that go with it)
   - One commit for Tuberia-S2 (WeightLoader changes)
   - The plan/state/brief files can ride along in a third "mission docs" commit.
3. `/mission-supervisor resume` — supervisor continues Acervo-S2 (push + tag + release v0.8.3), then dispatches Tuberia-S3.
4. After Tuberia-S3 verifies, run `/mission-supervisor brief` again to write the final iteration-01 brief.

If the user instead chooses to roll back (option B in §3.1), the rollback ritual from `commands/brief.md` applies. But that throws away ~1000 lines of working, tested code to fix two process gaps — not a good trade.
