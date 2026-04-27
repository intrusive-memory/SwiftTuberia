---
feature_name: OPERATION RIVETED PIPEWORK
mission_branch: mission/riveted-pipework/01
starting_point_commit: dcc1eec00e96125c18596b110cd73974e338ff4b
iteration: 1
---

# EXECUTION_PLAN.md — SwiftTuberia SwiftAcervo v2 Integration

**Source**: `REQUIREMENTS.md` (2026-04-20)
**Mission**: Complete SwiftTuberia's migration from SwiftAcervo v1 partial-registration integration to complete v2 abstracted-access integration with SHA-256 integrity verification and standardized CDN workflows.
**Refined**: 2026-04-20 — all 4 passes applied (atomicity, priority, parallelism, open questions).

---

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

---

## Mission Context

- **Upstream API**: `/Users/stovak/Projects/SwiftAcervo/API_REFERENCE.md`
- **Audit**: `AUDIT_FINDINGS.md` (Sorties T1–T5)
- **Architecture**: `requirements/` and `architecture/` (PROTOCOLS, PIPELINE, CATALOG, INFRASTRUCTURE, TESTING)
- **Verified external assumptions** (during refinement):
  - SwiftAcervo 0.7.2 ships the internal `withComponentAccess(_:in:perform:)` overload at `Sources/SwiftAcervo/AcervoManager.swift:455` (used by Sortie 6).
  - Only one production type conforms to `PipelineRecipe`: `MockPipelineRecipe` under `Tests/TuberiaGPUTests/Mocks/` — scopes Sortie 5's "update any recipe" task.

Status items 1–5 and 13 from the requirements Status Snapshot are ✅ DONE. This plan covers the remaining ❌ OPEN and ⚠️ PARTIAL items (rows 6–12).

---

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|--------------|
| SwiftTuberia-v2 | `.` (project root) | 8 | 1–4 | none |

Single-project mission. All sorties operate on the SwiftTuberia repo with intra-sortie dependencies tracked below.

---

## Sortie Dependency Graph

| Sortie | ID | Layer | Priority | Hard Dependencies | Notes |
|--------|----|-------|----------|-------------------|-------|
| 1 | REQ-T5 | 1 | 24.5 | none | `Package.swift` only — unblocks fresh-clone builds |
| 2 | REQ-T4 | 1 | 14.5 | none | `CatalogRegistration.swift` only; foundation for S6/S7 |
| 3 | REQ-PIPE-01 | 1 | 14.0 | none | `DiffusionPipeline.swift`; foundation for S4/S5 |
| 4 | REQ-PIPE-02 | 2 | 6.0 | S3 (REQ-PIPE-01) | same file as S3/S5 — must serialize |
| 5 | REQ-PIPE-03 | 2 | 6.5 | S3 (REQ-PIPE-01) | same file as S3/S4 — must serialize |
| 6 | REQ-INT-01 | 2 | 6.5 | S2 (REQ-T4) | needs populated checksums; new files only |
| 7 | REQ-CDN-01 | 2 | 7.0 | S2 (REQ-T4) | needs populated checksums; CI + helper |
| 8 | REQ-DOC-01 | 4 | 2.0 | S1–S7 | documentation sweep |

**Priority formula**: `(dep_depth × 3) + (foundation × 2) + (risk × 1) + (complexity × 0.5)`. Higher = earlier.

---

## Parallelism Structure

**Critical path**: S3 → S5 → S8 (length: 3 sorties). Mirrored by S2 → S6 → S8 and S3 → S4 → S8.

**Parallel execution groups**:

- **Group A (Layer 1 — fully parallel)**: S1, S2, S3 — independent files, no inter-dependencies.
  - S1 edits `Package.swift` + `AGENTS.md`.
  - S2 edits `CatalogRegistration.swift` + `CatalogRegistrationTests`.
  - S3 edits `DiffusionPipeline.swift` + pipeline tests.
- **Group B (Layer 2 — partial parallel)**: after Group A completes.
  - **Sub-group B1 (serialized on `DiffusionPipeline.swift`)**: S4 → S5 (or S5 → S4).
  - **Sub-group B2 (parallel with B1)**: S6 (new test files), S7 (CI workflow + new tool).
- **Group C (Layer 4 — sequential)**: S8 after everything else.

**Agent allocation**:

| Agent | Role | Handles |
|-------|------|---------|
| Supervising agent | All sorties with `xcodebuild` verification | S1, S2, S3, S4, S5, S6 |
| Sub-agent 1 | No-build file authoring | S7 workflow/helper authoring (supervisor runs the local verify) |
| Sub-agent 2 | Pure documentation | S8 |

**Build constraint**: Every sortie whose exit criteria invoke `xcodebuild build`/`xcodebuild test` is **supervising-agent-only**. Sub-agents may prepare code for S7/S8 but must NOT run builds.

**Maximum concurrency**: 3 agents in Group A (1 supervising sequential through S1→S2→S3, or supervising on one + sub-agents on non-build portions). Realistic throughput: supervising agent drives the serial build chain; sub-agents prep S7/S8 artifacts in parallel during Layer 2.

---

### Sortie 1: REQ-T5 — Bump `SwiftAcervo` Minimum Floor in `Package.swift`

**Priority**: 24.5 — highest foundation (7 sorties depend on v2 API; fresh-clone correctness).

**Entry criteria**:
- [ ] First-layer sortie — no prerequisites.
- [ ] `Package.resolved` currently pins `SwiftAcervo` at `0.7.2`.
- [ ] `Package.swift:23` currently reads `.package(..., from: "0.5.6")`.

**Tasks**:
1. Edit `Package.swift:23` — change `from: "0.5.6"` to `from: "0.7.2"` (smallest SwiftAcervo version exposing v2 API: `withComponentAccess`, `ComponentFile.sha256`, `ComponentHandle`, `Acervo.register`, `Acervo.ensureComponentReady`, `AcervoError.integrityCheckFailed`).
2. Run `xcodebuild -resolvePackageDependencies -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` and commit any resulting update to `Package.resolved`.
3. Reconcile `AGENTS.md` v0.3.0 changelog entry. **Chosen strategy**: add a **new** changelog entry documenting the floor bump to 0.7.2 (preserves history; do not silently rewrite 0.6.0). Reference this sortie ID (REQ-T5).

**Exit criteria**:
- [ ] `grep -E '"SwiftAcervo.*from: "0\.(7\.[2-9]|[89]\.|\d{2,})' Package.swift` returns a match (version ≥ 0.7.2).
- [ ] `rm -rf .build && xcodebuild -resolvePackageDependencies -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` succeeds; `grep 'version" : "0\.(7\.[2-9]|[89]\.|[1-9]\d' Package.resolved` returns a match.
- [ ] `xcodebuild build -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` exits 0.
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` exits 0.
- [ ] `AGENTS.md` contains a new changelog entry (NOT an edit of v0.3.0) mentioning `SwiftAcervo` and `0.7.2` (or higher): `grep -c 'SwiftAcervo.*0\.7\.2' AGENTS.md` ≥ 1.

---

### Sortie 2: REQ-T4 — Populate SHA-256 Checksums on All Catalog Descriptors

**Priority**: 14.5 — second-highest foundation (blocks S6 integrity tests and S7 CDN verification).

**Entry criteria**:
- [ ] First-layer sortie — no prerequisites (v2 API already available via `Package.resolved` 0.7.2).
- [ ] `Sources/TuberiaCatalog/Registration/CatalogRegistration.swift:12-38` currently contains 6 `ComponentFile(relativePath:)` entries with `sha256 = nil` and `expectedSizeBytes = nil`.

**Tasks**:
1. Pull canonical MLX artifacts from the CDN for each component (same slugs as `.github/workflows/ensure-model-cdn.yml`):
   - `intrusive-memory_t5-xxl-int4-mlx/`: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors`
   - `intrusive-memory_sdxl-vae-fp16-mlx/`: `config.json`, `model.safetensors`
2. Compute SHA-256 digests with `shasum -a 256 <file>` and sizes with `stat -f%z <file>`. Record results in a scratch file.
3. Rewrite every `ComponentFile(...)` entry in `CatalogRegistration.swift:12-38` to pass `ComponentFile(relativePath: …, expectedSizeBytes: <bytes>, sha256: "<64-char lowercase hex>")`.
4. Extend `Tests/TuberiaCatalogTests/CatalogRegistrationTests.swift` (or add a new file under the same target) with a test suite asserting for every registered `ComponentFile`: `sha256` is non-nil, `expectedSizeBytes > 0`, and `sha256` matches `^[0-9a-f]{64}$`.

**Exit criteria**:
- [ ] Every `ComponentFile` entry contains both keywords: `grep -cE 'sha256: *"[0-9a-f]{64}"' Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` returns `6` (one per file across both components).
- [ ] `grep -cE 'expectedSizeBytes: *[0-9]+' Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` returns `6`.
- [ ] Zero `ComponentFile` entries omit either argument: `grep -nE 'ComponentFile\([^)]*\)' Sources/TuberiaCatalog/Registration/CatalogRegistration.swift | grep -v 'sha256:' | wc -l | tr -d ' '` returns `0`.
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` exits 0, including the new assertion suite.

---

### Sortie 3: REQ-PIPE-01 — Call `Acervo.ensureComponentReady(_:)` Before Weight Loading

**Priority**: 14.0 — foundation for S4/S5 (all edit the same file); unblocks correct first-run behavior.

**Entry criteria**:
- [ ] First-layer sortie — no prerequisites.
- [ ] `Sources/Tuberia/Pipeline/DiffusionPipeline.swift:173-217` `loadModels(progress:)` does not currently call `Acervo.ensureComponentReady`.

**Tasks**:
1. In `loadModels(progress:)` `for segment…` loop, for each weighted segment with a non-nil `componentId`, call `try await Acervo.ensureComponentReady(componentId, progress: …)` **before** invoking `WeightLoader.load`.
2. Thread the existing `(Double, String)` progress callback through the `Acervo.ensureComponentReady` progress closure so download phases emit progress instead of a frozen bar.
3. **Chosen strategy for progress reporting**: fold download progress into the existing `progress(loadedCount/totalSegments, componentName)` tick stream — do **not** add a new `PipelineProgress` case (keeps the public enum stable; S8 does not need to document a new API). Inline-comment why.
4. Add a unit test (`Tests/TuberiaTests/PipelineLoadModelsTests.swift` or extension to an existing file) that verifies `ensureComponentReady` is invoked. Approach: inject a test double for the Acervo entry point via an existing seam if one exists, or expose a protocol-backed seam in this sortie. Cover both paths: not-on-disk (triggers download) and cached (no network).

**Exit criteria**:
- [ ] `grep -n 'ensureComponentReady' Sources/Tuberia/Pipeline/DiffusionPipeline.swift` returns ≥ 1 match inside the `loadModels(progress:)` function.
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` exits 0.
- [ ] New test file exists and asserts `ensureComponentReady` is called once per weighted segment with a non-nil `componentId`: `grep -r 'ensureComponentReady' Tests/` returns ≥ 1 match.
- [ ] `grep -n 'PipelineProgress' Sources/Tuberia/Pipeline/PipelineProgress.swift` shows no new cases added (public enum unchanged).

---

### Sortie 4: REQ-PIPE-02 — Wire `MemoryManager.hardValidate()` Into the Load Path

**Priority**: 6.0 — quality gate; depends on S3 for file serialization.

**Entry criteria**:
- [ ] REQ-PIPE-01 (Sortie 3) is COMPLETE — serializes edits to `DiffusionPipeline.swift`.
- [ ] `Sources/Tuberia/Infrastructure/MemoryManager.swift:81-99` declares `softCheck` and `hardValidate` but no production call site exists.

**Tasks**:
1. At entry to `loadModels(progress:)` in `DiffusionPipeline.swift:173-217`, compute peak memory requirement from `_memoryRequirement.peakMemoryBytes` and call `await MemoryManager.shared.hardValidate(requiredBytes: peak)`.
2. Define a pipeline-specific error variant `PipelineError.insufficientMemory(required: UInt64, available: UInt64)` in `Sources/Tuberia/Pipeline/PipelineError.swift`, and wrap any memory-manager error thrown from `hardValidate` in this variant at the call site.
3. **Chosen strategy**: single up-front `hardValidate(peakMemoryBytes)`. Phased-loading with `softCheck` is deferred (noted in Open Questions). Inline-comment the decision and reference this sortie.
4. Update `requirements/INFRASTRUCTURE.md:154-218` "Audit Checklist" to match the realized single-gate design (remove any phased-load claims; add `PipelineError.insufficientMemory` to the error table).
5. Add a unit test in `Tests/TuberiaTests/MemoryGuardTests.swift` that injects a stub `MemoryManager` (or overrides `availableMemory` for the test) with availability below `peakMemoryBytes` and asserts the pipeline throws `PipelineError.insufficientMemory` from `loadModels(progress:)`.

**Exit criteria**:
- [ ] `grep -nE 'hardValidate|MemoryManager\.shared' Sources/Tuberia/Pipeline/DiffusionPipeline.swift` returns ≥ 1 match inside `loadModels(progress:)`.
- [ ] `grep -nE 'case insufficientMemory' Sources/Tuberia/Pipeline/PipelineError.swift` returns a match.
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` exits 0.
- [ ] New unit test file `Tests/TuberiaTests/MemoryGuardTests.swift` exists and references `PipelineError.insufficientMemory`: `grep -l 'insufficientMemory' Tests/TuberiaTests/MemoryGuardTests.swift` returns the file path.
- [ ] `grep -n 'insufficientMemory\|hardValidate' requirements/INFRASTRUCTURE.md` returns ≥ 1 match.

---

### Sortie 5: REQ-PIPE-03 — Replace Positional `findComponentId(for:)` With Recipe Role Map

**Priority**: 6.5 — latent-bug fix; depends on S3 for file serialization.

**Entry criteria**:
- [ ] REQ-PIPE-01 (Sortie 3) is COMPLETE — serializes edits to `DiffusionPipeline.swift`.
- [ ] `Sources/Tuberia/Pipeline/DiffusionPipeline.swift:515-530` `findComponentId(for:)` still indexes `_allComponentIds` by hard-coded positional convention `[0]=encoder, [1]=backbone, [2]=decoder`.
- [ ] Only one production/test type conforms to `PipelineRecipe`: `Tests/TuberiaGPUTests/Mocks/MockPipelineRecipe.swift:10` — that is the only recipe to update.

**Tasks**:
1. Add `var componentIdFor: [PipelineRole: String] { get }` to `PipelineRecipe` in `Sources/Tuberia/Pipeline/PipelineRecipe.swift`. Provide a protocol-extension default implementation derived from zipping `allComponentIds` with `PipelineRole.allCases` iteration order — preserves today's positional convention for any conformer that doesn't override.
2. Replace `_allComponentIds: [String]` on `DiffusionPipeline` with `_componentIdByRole: [PipelineRole: String]`. Update the initializer that populates it to read `recipe.componentIdFor`.
3. Rewrite `findComponentId(for:)` as a dictionary lookup keyed by `PipelineRole`. Delete the positional indexing.
4. Update `Tests/TuberiaGPUTests/Mocks/MockPipelineRecipe.swift` to expose a `componentIdFor` override surface (default inherited is fine, but tests need to override).
5. Add a `PipelineAssemblyTests` case (existing file under `Tests/TuberiaTests/` or new `Tests/TuberiaTests/RecipeRoleMapTests.swift`) that builds a `MockPipelineRecipe` whose `componentIdFor` map is **reversed** relative to `allComponentIds` default, and asserts each segment loads its own weights (not its neighbor's) by observing which `componentId` strings reach the stubbed loader.

**Exit criteria**:
- [ ] `grep -n '_allComponentIds' Sources/Tuberia/Pipeline/DiffusionPipeline.swift` returns 0 matches.
- [ ] `grep -nE '\[0\]|\[1\]|\[2\]' Sources/Tuberia/Pipeline/DiffusionPipeline.swift` shows no positional indexing inside `findComponentId`.
- [ ] `grep -n 'componentIdFor' Sources/Tuberia/Pipeline/PipelineRecipe.swift` returns ≥ 1 match (protocol requirement + default).
- [ ] New reversed-order recipe test exists: `grep -rn 'reversed\|componentIdFor' Tests/TuberiaTests/` returns ≥ 1 match in a test file.
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` exits 0.

---

### Sortie 6: REQ-INT-01 — End-to-End `withComponentAccess` Integration Tests

**Priority**: 6.5 — closes integrity-path coverage gap; depends on S2.

**Entry criteria**:
- [ ] REQ-T4 (Sortie 2) is COMPLETE — production descriptors have SHA-256 checksums.
- [ ] No existing test exercises `AcervoManager.withComponentAccess`, `WeightLoader.load`, or integrity failure paths: `grep -r 'withComponentAccess' Tests/ | wc -l` returns 0.

**Tasks**:
1. Create `Tests/TuberiaCatalogTests/WeightLoaderIntegrationTests.swift`. Implement the **happy path** test: stage a temp App Group directory with a synthetic `.safetensors` file whose SHA-256 matches a test-only `ComponentDescriptor`, invoke `WeightLoader.load(componentId:…)` via the internal `withComponentAccess(_:in:perform:)` overload (verified available at `SwiftAcervo/Sources/SwiftAcervo/AcervoManager.swift:455`), assert `ModuleParameters` contains the expected key.
2. In `Tests/TuberiaCatalogTests/ComponentIntegrityTests.swift`, implement the **integrity failure** test: stage the same directory but corrupt a single byte in the safetensors file, assert load throws `AcervoError.integrityCheckFailed` (propagated as `PipelineError.weightLoadingFailed`).
3. Implement the **not-downloaded failure** test (in `WeightLoaderIntegrationTests.swift`): register a descriptor but stage no files, assert `PipelineError.modelNotDownloaded`.
4. Implement the **LoRA `withLocalAccess`** test (in `WeightLoaderIntegrationTests.swift`): stage a bare safetensors file on disk, call `LoRALoader.loadAdapterWeights(config: .init(localPath: url.path, …))`, assert non-empty params.
5. Verify all new tests honor `TESTING.md` rules: no real CDN weights, no network, no timed waits, use small synthetic tensors (< 1 MB each).

**Exit criteria**:
- [ ] `test -f Tests/TuberiaCatalogTests/WeightLoaderIntegrationTests.swift && test -f Tests/TuberiaCatalogTests/ComponentIntegrityTests.swift` both succeed.
- [ ] `grep -l 'integrityCheckFailed' Tests/TuberiaCatalogTests/ComponentIntegrityTests.swift` returns the file path.
- [ ] `grep -cE 'func test[A-Z]' Tests/TuberiaCatalogTests/WeightLoaderIntegrationTests.swift Tests/TuberiaCatalogTests/ComponentIntegrityTests.swift` returns ≥ 4 (happy, integrity, not-downloaded, LoRA).
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -only-testing:TuberiaCatalogTests/WeightLoaderIntegrationTests -only-testing:TuberiaCatalogTests/ComponentIntegrityTests` exits 0.
- [ ] Deliberately flipping a byte in the staged synthetic file during local run produces `AcervoError.integrityCheckFailed` (verified by the integrity test asserting the thrown error).
- [ ] Tests use no network: `grep -rnE 'URLSession|URLRequest|http(s)?://' Tests/TuberiaCatalogTests/WeightLoaderIntegrationTests.swift Tests/TuberiaCatalogTests/ComponentIntegrityTests.swift` returns 0 matches.

---

### Sortie 7: REQ-CDN-01 — Verify Uploaded Manifest Matches Repository Descriptors

**Priority**: 7.0 — guards checksum drift; depends on S2.

**Entry criteria**:
- [ ] REQ-T4 (Sortie 2) is COMPLETE — `CatalogRegistration.swift` contains populated `sha256` and `expectedSizeBytes` per file.
- [ ] `.github/workflows/ensure-model-cdn.yml` currently runs `acervo manifest create` + `acervo upload` with a `Verify upload` step but does not cross-check manifest digests against source-of-truth descriptors.

**Tasks**:
1. **Chosen helper form**: create a new SwiftPM executable target at `Tools/VerifyComponentManifest/main.swift` (or similar). Rationale: a standalone executable is invokable from CI without spinning up the XCTest runtime; a dumping test would couple CI verification to the test target. Add the target to `Package.swift` (plugin-free, depends on `TuberiaCatalog`).
2. The executable reads a downloaded `manifest.json` path from argv, loads the registered `ComponentDescriptor` set from `CatalogRegistration`, and compares `{path, size, sha256}` per file. Prints diff-style mismatches to stderr and exits non-zero on any divergence; exits 0 on full match.
3. Add a `Verify manifest matches source` step to `.github/workflows/ensure-model-cdn.yml` that runs **after** `Verify upload`. The step downloads the uploaded `manifest.json` and invokes the executable from task 1.
4. On any mismatch in `sha256` or `expectedSizeBytes`, the job fails with a clear diff-style error message naming the component, path, expected vs actual values (the executable already emits this; the CI step just surfaces the exit code).
5. Document the verification contract via inline comments in `.github/workflows/ensure-model-cdn.yml`.

**Exit criteria**:
- [ ] `test -f Tools/VerifyComponentManifest/main.swift` succeeds (or the chosen target path documented in Package.swift).
- [ ] `grep -n 'VerifyComponentManifest\|verify_component_manifest\|verify-manifest' Package.swift` returns ≥ 1 match (target registered).
- [ ] `xcodebuild build -scheme VerifyComponentManifest -destination 'platform=macOS,arch=arm64'` exits 0 (the new target builds).
- [ ] `grep -nE 'Verify manifest matches source|VerifyComponentManifest' .github/workflows/ensure-model-cdn.yml` returns ≥ 1 match AFTER the existing `Verify upload` step (line-number ordering check).
- [ ] Local divergence test: after editing one hex digit in `CatalogRegistration.swift`, running the executable against the unedited manifest exits non-zero; reverting the edit makes it exit 0. (Documented in the sortie log; reviewer reproduces.)
- [ ] YAML remains parse-valid: `python3 -c 'import yaml,sys; yaml.safe_load(open(".github/workflows/ensure-model-cdn.yml"))'` exits 0.

---

### Sortie 8: REQ-DOC-01 — Update AGENTS.md / Architecture Docs for v2 Completion

**Priority**: 2.0 — final documentation sweep; runs only after S1–S7 ship.

**Entry criteria**:
- [ ] Sorties 1–7 are COMPLETE and merged.
- [ ] `AGENTS.md` "Recent Changes" and `requirements/INFRASTRUCTURE.md:209-218` "Audit Checklist" currently contain stale claims.

**Tasks**:
1. Add a new changelog entry (next available version, e.g. v0.3.7) to `AGENTS.md` summarizing: SwiftAcervo floor → 0.7.2 (S1), per-file SHA-256 checksums (S2), `ensureComponentReady` wired in (S3), `hardValidate` + `PipelineError.insufficientMemory` (S4), role-map `findComponentId` (S5), integration tests (S6), CDN manifest verification (S7). One bullet per sortie, each referencing its sortie ID.
2. Audit `requirements/INFRASTRUCTURE.md:209-218` Audit Checklist — for every `[x]` item, confirm it is truly implemented; convert `[x]`→`[ ]` if still incomplete, or keep `[x]` with a parenthetical `(REQ-XXX)` link to the sortie/commit that satisfied it.
3. Add `- [REQUIREMENTS.md](REQUIREMENTS.md) — active mission scope` to `AGENTS.md`'s top-level "Architecture" section (or equivalent).
4. Reconcile `architecture/PROTOCOLS.md` and `architecture/INFRASTRUCTURE.md` load-path descriptions with the implementation after S3/S4/S5. If they diverge, update the doc; if they already match, no edit needed.
5. Mark `REQUIREMENTS.md` Status Snapshot rows 6–12 as ✅ DONE, each row's Evidence column linking to the satisfying sortie/commit.

**Exit criteria**:
- [ ] `grep -nE 'v0\.3\.[7-9]|v0\.[4-9]' AGENTS.md` returns ≥ 1 match (new changelog entry exists).
- [ ] New changelog entry mentions all 7 sortie IDs: `grep -oE 'REQ-(T[45]|PIPE-0[123]|INT-01|CDN-01)' AGENTS.md | sort -u | wc -l | tr -d ' '` returns `7`.
- [ ] `grep -n 'REQUIREMENTS\.md' AGENTS.md` returns ≥ 1 match (cross-link).
- [ ] `grep -cE '^\| *[6-9] \|.*✅ DONE' REQUIREMENTS.md` returns `7` (rows 6 through 12 all marked DONE).
- [ ] `grep -cE '^- \[x\]' requirements/INFRASTRUCTURE.md` count matches actual implemented items (manual attestation by sortie author in commit message: "verified every [x] in INFRASTRUCTURE.md Audit Checklist").
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` exits 0 (no regression from prose-only edits).

---

## Master Acceptance (from REQUIREMENTS.md § Definition of Done)

Mission complete when all of the following hold:

- [ ] All rows in the `REQUIREMENTS.md` Status Snapshot read ✅ DONE.
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'` passes on a clean clone.
- [ ] `xcodebuild test …` with a locally corrupted component file surfaces `AcervoError.integrityCheckFailed` (propagated as `PipelineError.weightLoadingFailed`).
- [ ] Fresh `swift package reset && xcodebuild build` resolves SwiftAcervo ≥ 0.7.2.
- [ ] `ensure-model-cdn.yml` workflow's manifest-verification step fails on a deliberate checksum mismatch.
- [ ] Master index `/Users/stovak/Projects/REQUIREMENTS.md` SwiftTuberia row moves from 🔴 RED 0% to 🟢 GREEN 100%; Sorties 2.1–2.3 + T1–T5 all marked 🟢 COMPLETE.

---

## Open Questions & Missing Documentation

Issues surfaced during Pass 4 refinement. Items marked **RESOLVED** were auto-fixed in the sortie bodies above; items marked **NOTED** are non-blocking design decisions already captured inline; items marked **DEFERRED** are out of scope for this mission.

| # | Sortie | Issue Type | Description | Resolution |
|---|--------|-----------|-------------|------------|
| 1 | S1 | Design choice | AGENTS.md v0.3.0 changelog says "0.6.0" — rewrite vs append new entry | **RESOLVED**: append new entry (preserves history). Exit criterion enforces via grep. |
| 2 | S2 | Flawed exit criterion | Original `grep 'ComponentFile(relativePath:'` returned-zero check is false (new form also matches) | **RESOLVED**: replaced with positive-assertion grep counting `sha256:` and `expectedSizeBytes:` entries. |
| 3 | S3 | Design choice | New `PipelineProgress.downloading(…)` case vs fold into existing tick stream | **RESOLVED**: fold into existing stream; public enum unchanged. Exit criterion enforces no new case. |
| 4 | S3 | Vague exit criterion | "Behavior satisfies PROTOCOLS.md step sequencing" not machine-verifiable | **RESOLVED**: replaced with grep + test file existence + test assertion check. |
| 5 | S4 | Design choice | `hardValidate(peak)` once vs `softCheck` + phased `hardValidate` per phase | **RESOLVED**: single up-front `hardValidate`. Phased load **DEFERRED** to a future mission if real peak-vs-phase divergence appears. |
| 6 | S4 | Error-type ambiguity | Task 2 referenced `PipelineError.insufficientMemory` without defining it | **RESOLVED**: sortie now explicitly defines the case in `PipelineError.swift` and tests for it. |
| 7 | S5 | Scope ambiguity | "Update any recipe that relied on positional ordering" — how many recipes exist? | **RESOLVED**: verified exactly one conformer (`MockPipelineRecipe`). Entry criteria record this. |
| 8 | S6 | External assumption | Assumes internal `withComponentAccess(_:in:perform:)` overload exists in 0.7.2 | **RESOLVED**: verified at `SwiftAcervo/Sources/SwiftAcervo/AcervoManager.swift:455`. |
| 9 | S7 | Design choice | Executable target vs descriptor-dumping test | **RESOLVED**: executable target chosen (decouples CI from test runtime). |
| 10 | S7 | Conditional tooling | "actionlint or yamllint clean **if such tooling is in use**" — fallback undefined | **RESOLVED**: replaced with unconditional `python3 -c 'import yaml; yaml.safe_load(…)'` parse check. |
| 11 | S8 | Subjective exit | "A fresh reader can reconstruct the v2 integration state… without running grep" | **RESOLVED**: replaced with objective checks (sortie-ID coverage, cross-link presence, DONE-row count). |
| 12 | S8 | Manual attestation | INFRASTRUCTURE.md checklist audit can't be fully machine-verified | **NOTED**: reviewer must attest in commit message; remaining exit criteria are machine-checkable. |
| 13 | All | Non-blocking items | LoRA descriptors, plugin backbones, CLIP/DDPM, renderers | **DEFERRED** — explicitly captured in REQUIREMENTS.md § Non-Blocking / Deferred Items. |

**Blocking issues remaining**: 0.

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 1 |
| Total sorties | 8 |
| Dependency structure | 4 layers (1, 2, 3, 4); Layer 1 fully parallel; Layer 2 partial-parallel (S4/S5 serialize on `DiffusionPipeline.swift`) |
| Critical path | S3 → S5 → S8 (length 3); equivalent: S2 → S6 → S8, S3 → S4 → S8 |
| Max concurrency | 3 sorties in Layer 1 + Layer 2 work (supervising + up to 2 sub-agents on non-build portions) |
| Build-constrained sorties | S1, S2, S3, S4, S5, S6 (supervising-agent-only) |
| Sub-agent-eligible | S7 (authoring), S8 (prose) |
| Files primarily touched | `Package.swift`, `Sources/TuberiaCatalog/Registration/CatalogRegistration.swift`, `Sources/Tuberia/Pipeline/{DiffusionPipeline,PipelineRecipe,PipelineError}.swift`, `Sources/Tuberia/Infrastructure/MemoryManager.swift`, `Tests/TuberiaCatalogTests/**`, `Tests/TuberiaTests/**`, `Tools/VerifyComponentManifest/**`, `.github/workflows/ensure-model-cdn.yml`, `AGENTS.md`, `REQUIREMENTS.md`, `requirements/INFRASTRUCTURE.md`, `architecture/**` |

---

## Refinement Pass Results

| Pass | Status | Changes |
|------|--------|---------|
| 1. Atomicity & Testability | ✓ PASS | 0 splits, 0 merges; 4 vague exit criteria rewritten as machine-verifiable grep/file-exists/exit-code checks (S2 grep bug, S3 protocol-compliance claim, S7 conditional tooling, S8 subjective reader test) |
| 2. Prioritization | ✓ PASS | Priority scores added to all 8 sorties; existing 1→8 numbering preserved (priorities align with dependency order; no renumber needed) |
| 3. Parallelism | ✓ PASS | Critical path = 3 sorties; Group A (Layer 1) fully parallel; Group B partial; build-constraint map added (6 supervising, 2 sub-agent-eligible) |
| 4. Open Questions & Vague Criteria | ✓ PASS | 12 issues catalogued — 11 auto-resolved inline, 1 requires reviewer attestation (S8 checklist audit), 0 blocking |

**VERDICT**: ✓ Plan is ready to execute.

**Next step**: `/mission-supervisor start`
