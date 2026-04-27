---
feature_name: OPERATION VANISHING MANIFEST
starting_point_commit: 36887b979810af7ee30d5f8fdd49d6322c52b656
mission_branch: mission/vanishing-manifest/01
iteration: 1
---

# EXECUTION_PLAN.md — SwiftTuberia SwiftAcervo V2 Compliance

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

---

## Overview

**Goal**: SwiftTuberia never deals with files. Acervo provides all model access.

All 8 requirements from `requirements/ACERVO_V2_COMPLIANCE.md` are addressed across two repos:
- **SwiftAcervo** (`../SwiftAcervo`): API surface additions that SwiftTuberia depends on
- **SwiftTuberia** (`.`): Compliance changes consuming those APIs

Requirements TUBERIA-V2-01, V2-02, V2-03 and ACERVO-V2-01 are independent and run in Layer 1 (all builds on supervising agent; ACERVO-V2-01 release on sub-agent in parallel with Tuberia-S1/S2).
TUBERIA-V2-04 is blocked by SwiftAcervo v0.8.3 being released and runs in Layer 2.
ACERVO-V2-02 is optional and runs last.

---

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|-------------|
| SwiftAcervo | `../SwiftAcervo` | 3 (S3 optional) | 1–2 | S2 ← S1; S3 ← S1 (optional) |
| SwiftTuberia | `.` | 3 | 1–2 | S3 ← SwiftAcervo-S2 (release), S3 ← S1, S3 ← S2 |

---

## Parallelism Structure

**Critical Path**: SwiftAcervo-S1 → SwiftTuberia-S1 → SwiftTuberia-S2 → SwiftTuberia-S3 (4 sorties; supervising agent — Tuberia-S1/S2 have no data dependency on Acervo-S1 but serialize behind it on the supervising agent queue)

**Parallel Execution Groups**:
- **Group 1** (supervising agent only):
  - SwiftAcervo-S1 — Add rootDirectoryURL (build in `../SwiftAcervo`) — **SUPERVISING AGENT ONLY**
- **Group 2** (parallel after Group 1 completes):
  - SwiftAcervo-S2 — Release v0.8.3 (no build: git/gh ops only) — **Sub-agent (Agent 2)**
  - SwiftTuberia-S1 — Bare descriptors (build in `.`) — **SUPERVISING AGENT ONLY**
- **Group 3** (supervising agent, after SwiftTuberia-S1; sub-agent Group 2 likely done by now):
  - SwiftTuberia-S2 — Dead code removal (build in `.`) — **SUPERVISING AGENT ONLY**
- **Group 4** (after both Group 2 chains are verified complete):
  - SwiftTuberia-S3 — rootDirectoryURL + docstring (build in `.`) — **SUPERVISING AGENT ONLY**
- **Group 5** (optional, any time after Group 1):
  - SwiftAcervo-S3 — estimatedSizeBytes hint (build in `../SwiftAcervo`) — **SUPERVISING AGENT ONLY**

**Agent Constraints**:
- **Supervising agent**: Handles ALL sorties with build steps (all SwiftAcervo and SwiftTuberia build sorties). Serial within each repo to avoid xcodebuild conflicts.
- **Sub-agent (Agent 2)**: SwiftAcervo-S2 release only (no build; git tag + gh release ops). Runs concurrently with Group 2 supervising work.

**Missed Opportunities**:
- SwiftAcervo-S1 and SwiftTuberia-S1/S2 could theoretically run on separate agents (different repos, no shared build artifacts) but the "no build on sub-agents" constraint prevents this. All build sorties serialize on the supervising agent.

---

## Work Unit: SwiftAcervo

### Sortie 1: Add rootDirectoryURL to ComponentHandle

**Priority**: 7.5 — Foundational: unblocks SwiftAcervo-S2 (release) which gates SwiftTuberia-S3; establishes new public API surface; highest dependency depth in the mission.

**Entry criteria**:
- [ ] First sortie for SwiftAcervo — no prerequisites
- [ ] `../SwiftAcervo/Sources/SwiftAcervo/ComponentHandle.swift` is readable

**Tasks**:
1. Open `ComponentHandle.swift` and locate `internal var baseDirectory: URL` (or equivalent internal property)
2. Add `public var rootDirectoryURL: URL { baseDirectory }` immediately after it
3. Verify `LocalHandle` has a matching accessor (expected: `public let rootURL: URL`) — document any naming mismatch in a comment if found
4. Run `xcodebuild test -scheme SwiftAcervo-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO` from `../SwiftAcervo`

**Exit criteria**:
- [ ] `grep -n 'public var rootDirectoryURL' ../SwiftAcervo/Sources/SwiftAcervo/ComponentHandle.swift` returns 1 match
- [ ] `xcodebuild test` exits 0 (all existing ComponentHandle tests pass)
- [ ] No other files in SwiftAcervo are modified

---

### Sortie 2: Commit, push, and release SwiftAcervo v0.8.3

**Priority**: 7.0 — Critical-path blocker: SwiftTuberia-S3 cannot call `xcodebuild -resolvePackageDependencies` until a ≥0.8.3 tag exists on `intrusive-memory/SwiftAcervo`. No build step — runs on sub-agent concurrently with SwiftTuberia-S1/S2.

**Entry criteria**:
- [ ] SwiftAcervo-S1 is COMPLETED (`rootDirectoryURL` added and all tests passing)
- [ ] `git -C ../SwiftAcervo status` shows clean working tree (no uncommitted changes)

**Tasks**:
1. From `../SwiftAcervo`, run `/ship-swift-library` (or equivalent: bump version to 0.8.3, commit, push to main, create and push tag v0.8.3, create GitHub release)
2. Verify the GitHub release is published: `gh release view v0.8.3 --repo intrusive-memory/SwiftAcervo`

**Exit criteria**:
- [ ] `git -C ../SwiftAcervo tag -l 'v0.8.3'` returns `v0.8.3`
- [ ] `gh release view v0.8.3 --repo intrusive-memory/SwiftAcervo` exits 0

---

### Sortie 3: Allow estimatedSizeBytes hint on bare ComponentDescriptor [OPTIONAL]

**Priority**: 1.25 — Optional; no other sortie depends on this; zero dependency depth; low risk.

**Entry criteria**:
- [ ] SwiftAcervo-S1 is COMPLETED (rootDirectoryURL shipped)
- [ ] `../SwiftAcervo/Sources/SwiftAcervo/ComponentDescriptor.swift` is readable

**Tasks**:
1. Locate the bare `ComponentDescriptor` initializer (the one without a `files:` parameter)
2. Add `estimatedSizeBytes: Int64? = nil` as an optional parameter
3. Store the value in `self._estimatedSizeBytes` (or the property that backs `estimatedSizeBytes`)
4. Ensure the `estimatedSizeBytes` computed property returns the hint before hydration, manifest sum after
5. Run `xcodebuild test -scheme SwiftAcervo-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO`

**Exit criteria**:
- [ ] `grep -n 'estimatedSizeBytes.*Int64.*nil' ../SwiftAcervo/Sources/SwiftAcervo/ComponentDescriptor.swift` returns ≥ 1 match
- [ ] `xcodebuild test` exits 0
- [ ] Existing full-initializer behavior is unchanged (covered by `xcodebuild test` above)

---

## Work Unit: SwiftTuberia

### Sortie 1: Bare descriptors + deprecation in CatalogRegistration.swift

**Priority**: 5.75 — Dependency depth 1 (SwiftTuberia-S3 entry criterion requires S1 completed); medium risk (removes live code with test coverage).

**Entry criteria**:
- [ ] First sortie for SwiftTuberia — no prerequisites
- [ ] `Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` is readable

**Tasks**:
1. Delete the `t5XXLEncoderRequiredFiles` array literal (lines ~14–23 per requirements)
2. Delete the `sdxlVAEDecoderRequiredFiles` array literal
3. Replace the `t5XXLEncoderComponentDescriptor` full initializer with the bare initializer form (keeping `id`, `type`, `displayName`, `repoId`, `minimumMemoryBytes`, `metadata`; removing `files:` and `estimatedSizeBytes:`)
4. Replace the `sdxlVAEDecoderComponentDescriptor` full initializer with bare form
5. Add `@available(*, deprecated, message: "Use ComponentReadinessService (with progress) or Acervo.ensureComponentReady directly.")` to `CatalogRegistration.ensureComponentReady(_:)` (line ~139)
6. Run `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO`

**Exit criteria**:
- [ ] `grep -c 'ComponentFile(' Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` returns `0`
- [ ] `grep -c 'sha256:' Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` returns `0`
- [ ] `grep -c 'files:' Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` returns `0`
- [ ] `grep -n '@available.*deprecated' Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` returns 1 match on the `ensureComponentReady` method
- [ ] `xcodebuild test` exits 0

---

### Sortie 2: Dead code removal + typed error handling in WeightLoader.swift

**Priority**: 5.75 — Dependency depth 1 (SwiftTuberia-S3 entry criterion requires S2 completed); higher risk than S1 (error handling refactor touches runtime behavior paths).

**Entry criteria**:
- [ ] First sortie for this file — no prerequisites
- [ ] `Sources/Tuberia/Infrastructure/WeightLoader.swift` is readable
- [ ] Read `../SwiftAcervo/Sources/SwiftAcervo/AcervoError.swift` (or search `../SwiftAcervo` for `enum AcervoError`) to confirm exact case names before writing pattern-match code

**Tasks**:
1. Delete the `canEnumerateDirectory(_:)` method body (approximately lines 168–177)
2. Delete the `findSafetensorsFiles(in:)` method body (approximately lines 187–239)
3. Verify no callers exist: `grep -rn 'canEnumerateDirectory\|findSafetensorsFiles' Sources/` must return 0
4. In `WeightLoader.load()`, replace the string-inspection catch block (lines ~98–108) with typed `AcervoError` pattern matching: catch `componentNotDownloaded`, `componentNotRegistered`, `componentNotHydrated` → rethrow as `PipelineError.modelNotDownloaded`; catch `integrityCheckFailed` → rethrow as `PipelineError.weightLoadingFailed` with detail; re-catch `PipelineError` passthrough; fallback catch
5. Apply the same typed pattern matching to `loadFromPath()`'s catch block (lines ~151–158)
6. Run `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO`

**Exit criteria**:
- [ ] `grep -n 'findSafetensorsFiles\|canEnumerateDirectory\|FileManager' Sources/Tuberia/Infrastructure/WeightLoader.swift` returns 0 matches
- [ ] `grep -n 'contains("Not\|contains("not\|contains("invalid' Sources/Tuberia/Infrastructure/WeightLoader.swift` returns 0 matches
- [ ] `grep -n 'AcervoError\.' Sources/Tuberia/Infrastructure/WeightLoader.swift` returns ≥ 3 matches
- [ ] `xcodebuild test` exits 0

---

### Sortie 3: rootDirectoryURL + precondition docstring in T5XXLEncoder.swift

**Priority**: 2.5 — Blocked by SwiftAcervo-S2 (release) and SwiftTuberia-S1/S2; integrating sortie that closes the mission.

**Entry criteria**:
- [ ] SwiftAcervo-S2 is COMPLETED (v0.8.3 released and tag pushed to `intrusive-memory/SwiftAcervo`)
- [ ] SwiftTuberia-S1 is COMPLETED
- [ ] SwiftTuberia-S2 is COMPLETED
- [ ] `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` is readable

**Tasks**:
1. Run `xcodebuild -resolvePackageDependencies -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` to pull SwiftAcervo ≥ 0.8.3 — verify `grep -A3 '"SwiftAcervo"' Package.resolved` shows version ≥ 0.8.3
2. In `T5XXLEncoder.loadTokenizer()` (lines ~67–72): replace `let tokenizerURL = try handle.url(matching: "tokenizer.json"); return tokenizerURL.deletingLastPathComponent()` with `return handle.rootDirectoryURL`
3. Add precondition docstring to `loadTokenizer()`: must mention "ensureComponentReady" and "precondition" and note that `DiffusionPipeline.loadModels()` satisfies the precondition
4. Run `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO`

**Exit criteria**:
- [ ] `grep -n 'deletingLastPathComponent' Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` returns 0 matches
- [ ] `grep -n 'rootDirectoryURL' Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` returns ≥ 1 match
- [ ] `grep -n 'ensureComponentReady' Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` returns ≥ 1 match AND `grep -En '^\s*///.*[Pp]recondition' Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` returns ≥ 1 match
- [ ] `xcodebuild test` exits 0 (TuberiaCatalogTests tokenizer loads correctly)

---

## Open Questions & Missing Documentation

### Resolved Items (auto-fixed during refinement)

| Sortie | Issue Type | Description | Resolution |
|--------|-----------|-------------|------------|
| SwiftAcervo → SwiftTuberia boundary | Missing step | SwiftTuberia consumes SwiftAcervo as a remote package (`upToNextMajor(from: "0.8.2")`). Changes from Acervo-S1 must be tagged and released before `xcodebuild -resolvePackageDependencies` can fetch them. Original plan had no release sortie. | AUTO-FIXED: Added SwiftAcervo-S2 (release sortie) |
| SwiftAcervo-S3 (was S2) exit criterion 1 | Vague criterion | "ComponentDescriptor bare initializer accepts `estimatedSizeBytes: Int64? = nil`" is not machine-verifiable | AUTO-FIXED: Replaced with `grep -n 'estimatedSizeBytes.*Int64.*nil'` returning ≥ 1 match |
| SwiftTuberia-S3 exit criterion 3 | Vague criterion | "Docstring on `loadTokenizer` contains 'ensureComponentReady' and 'precondition'" not machine-verifiable | AUTO-FIXED: Replaced with two independent grep commands |
| SwiftTuberia-S2 entry criteria | Missing context | AcervoError case names required for typed pattern matching but source location unspecified | AUTO-FIXED: Added entry criterion to read `../SwiftAcervo` AcervoError before writing catch blocks |
| SwiftTuberia-S3 Task 1 | Vague instruction | "swift package update SwiftAcervo (via xcodebuild resolve)" — `swift package update` is excluded per CLAUDE.md (only `swift build` and `swift test` are forbidden, but intent was xcodebuild) | AUTO-FIXED: Changed to `xcodebuild -resolvePackageDependencies` with exact flags |
| SwiftAcervo-S2 exit criterion 3 | Sub-agent constraint violation + race condition | Exit criterion ran `xcodebuild -resolvePackageDependencies` on SwiftTuberia from the sub-agent. Violates "sub-agent: no build operations" constraint. Also races with supervising agent modifying Package.resolved concurrently during Tuberia-S1/S2. Package.resolved update is S3's responsibility (Task 1). | AUTO-FIXED: Removed the criterion from S2; two remaining criteria (git tag + gh release) fully verify the release |
| SwiftTuberia-S3 exit criterion 3 | Vague criterion | `grep -n 'precondition'` would match any `precondition()` assertion in code, not just the docstring — false positive risk | AUTO-FIXED: Tightened to `grep -En '^\s*///.*[Pp]recondition'` which requires the word to appear in a Swift doc-comment line |
| SwiftTuberia-S1 and S2 priority notes | Inaccurate annotation | Priority notes said "no other sortie depends on this" — factually wrong; SwiftTuberia-S3 entry criteria explicitly require both S1 and S2 COMPLETED. Scores were 2.5/2.75 (dep_depth=0 assumed); corrected to 5.75 each (dep_depth=1, formula: 3+0+2+0.83=5.83) | AUTO-FIXED: Updated priority notes and scores to reflect dep_depth=1 |

**Auto-Fixed**: 8
**Requires Manual Review**: 0

---

## Definition of Done (Full Mission)

All of the following must hold simultaneously:

1. `grep -rn 'ComponentFile\|findSafetensorsFiles\|canEnumerateDirectory\|FileManager' Sources/` returns 0 matches
2. `grep -rn 'contains("Not\|contains("not' Sources/Tuberia/Infrastructure/WeightLoader.swift` returns 0 matches
3. `grep -rn 'deletingLastPathComponent' Sources/TuberiaCatalog/` returns 0 matches
4. `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO` exits 0
5. All `ComponentHandle` and `LocalHandle` accesses in SwiftTuberia go through `.url(for:)`, `.url(matching:)`, `.urls(matching:)`, or `.rootDirectoryURL`

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 2 (SwiftAcervo, SwiftTuberia) |
| Total sorties | 6 (3 + 3; +1 release sortie added during refinement) |
| Dependency structure | SwiftAcervo-S1 → S2 (release) → SwiftTuberia-S3; Tuberia-S1/S2 parallel with Acervo-S2 on supervising agent |
| Optional sorties | 1 (SwiftAcervo-S3) |
| Critical path | 4 sorties (Acervo-S1 → Tuberia-S1 → Tuberia-S2 → Tuberia-S3; supervising agent) |
| Parallelism | 1 supervising agent + 1 sub-agent (Acervo-S2 release runs while supervising agent does Tuberia-S1/S2) |
