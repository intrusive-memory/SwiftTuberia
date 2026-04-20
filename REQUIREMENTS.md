---
title: "SwiftTuberia — SwiftAcervo v2 Integration Requirements"
date: 2026-04-20
status: "ACTIVE — partially implemented"
master_index: "/Users/stovak/Projects/REQUIREMENTS.md"
audit_source: "AUDIT_FINDINGS.md (Sorties T1–T5)"
---

# SwiftTuberia — SwiftAcervo v2 Integration Requirements

**Mission**: Complete SwiftTuberia's migration from SwiftAcervo v1 partial-registration
integration to **complete v2 abstracted-access integration** with SHA-256 integrity
verification and standardized CDN workflows.

**Upstream API reference**: `/Users/stovak/Projects/SwiftAcervo/API_REFERENCE.md`
(`AcervoManager.withComponentAccess`, `Acervo.register`, `ComponentFile.sha256`,
`AcervoError.integrityCheckFailed`, etc.)

**Architectural context**: See `docs/incomplete/REQUIREMENTS_ARCHITECTURE.md` (plumbing
system overview), `requirements/` (pipe segment contracts, pipeline, catalog), and
`architecture/` (companion architecture notes).

---

## Status Snapshot (2026-04-20)

| # | Area | AUDIT Sortie | Status | Evidence |
|---|---|---|---|---|
| 1 | Remove duplicate `ComponentDescriptor` type | T1 | ✅ DONE | `Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` imports and uses `SwiftAcervo.ComponentDescriptor`; no shadow type defined. |
| 2 | Migrate to `Acervo.register()` | T2 | ✅ DONE | `CatalogRegistration.swift:85-90` — `Acervo.register([t5XXL, sdxlVAE])` fires via module-level `let`. |
| 3 | `WeightLoader` on v2 API | T3 | ✅ DONE | `Sources/Tuberia/Infrastructure/WeightLoader.swift:40,132` — uses `AcervoManager.shared.withComponentAccess(...)` and `withLocalAccess(...)`. No `withModelAccess` calls remain. |
| 4 | Tokenizer resolution via v2 API | T3b | ✅ DONE | `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift:62` — `AcervoManager.shared.withComponentAccess(componentId) { handle in handle.url(matching: "tokenizer.json") }`. |
| 5 | LoRA adapters via `withLocalAccess` | T3c | ✅ DONE | `Sources/Tuberia/Pipeline/LoRALoader.swift` → `WeightLoader.loadFromPath` → `AcervoManager.shared.withLocalAccess`. |
| 6 | SHA-256 checksums on descriptors | T4 | ✅ DONE | REQ-T4 (corrected, S9) — Acervo v2 CDN manifest owns file integrity; Tuberia descriptors carry no caller-side sha256 or expectedSizeBytes. SwiftAcervo's per-file SHA-256 verification fires on `withComponentAccess`; no second source of truth in CatalogRegistration.swift. |
| 7 | Bump `SwiftAcervo` dependency floor | T5 | ✅ DONE | REQ-T5 (S1 `0aa8fcf`) — `Package.swift` now declares `from: "0.7.2"`, eliminating the v1-regression risk on fresh resolutions. AGENTS.md v0.3.7 and v0.3.8 changelog entries document the floor bump. |
| 8 | `Acervo.ensureComponentReady` in load path | — | ✅ DONE | REQ-PIPE-01 (S3 `de8212c`) — `DiffusionPipeline.loadModels(progress:)` now calls `ensureComponentReady` per segment via `ComponentReadinessService` seam before `WeightLoader.load`. First-run cache misses auto-download instead of throwing `componentNotDownloaded`. |
| 9 | `MemoryManager` pre-load validation | — | ✅ DONE | REQ-PIPE-02 (S4 `0c58bf5`) — single up-front `hardValidate(peakMemoryBytes)` via `memoryGate` seam at entry to `loadModels(progress:)`; throws `PipelineError.insufficientMemory(required:available:component:)` on budget exhaustion. `MemoryGuardTests.swift` verifies failure and pass-through paths. |
| 10 | Recipe→componentId lookup correctness | — | ✅ DONE | REQ-PIPE-03 (S5 `405168e`) — `_allComponentIds` replaced with `_componentIdByRole: [PipelineRole: String]`; `findComponentId(for:)` is now a role-keyed dictionary lookup. `PipelineRecipe` gains `componentIdFor` with a default that preserves the previous positional convention. |
| 11 | End-to-end v2 integration test | — | ✅ DONE | REQ-INT-01 (S6 `bf761d0`) — `WeightLoaderIntegrationTests.swift` and `ComponentIntegrityTests.swift` added to `TuberiaCatalogTests`: happy path, integrity failure (`AcervoError.integrityCheckFailed`), not-downloaded failure, and LoRA `withLocalAccess`. All tests use synthetic tensors; no network. |
| 12 | CDN upload workflow checksum verification | — | ✅ DONE | REQ-CDN-01 (corrected, S9) — `VerifyComponentManifest` removed; SwiftAcervo's CDN-manifest download + per-file integrity-check path is authoritative. `ensure-model-cdn.yml` verifies the manifest exists on the CDN after upload; file-level integrity is SwiftAcervo's responsibility at download time. |
| 13 | This `REQUIREMENTS.md` exists | Master 2.2 | ✅ DONE (this file) | Master index previously marked 🔴 BLOCKED. |

---

## Outstanding Work (Ordered)

The status snapshot is authoritative; the sortie entries below expand acceptance criteria
for every ❌/⚠️ row. Each sortie is atomic — complete in order unless parallelization is
explicitly called out.

### REQ-T4: Populate SHA-256 Checksums on All Catalog Descriptors

**Priority**: 🔴 CRITICAL — blocks integrity verification of every downloaded weight.

**Files**:
- `Sources/TuberiaCatalog/Registration/CatalogRegistration.swift:12-38`

**Current state (as of REQ-T4 completion)**: All 11 `ComponentFile` entries now carry
`sha256:` and `expectedSizeBytes:`. The original plan assumed 6 entries (4 for T5-XXL, 2
for SDXL VAE), but the actual T5-XXL descriptor uses 5 sharded safetensors files + 4
metadata files (config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json)
= 9 T5 files + 2 VAE files = 11 total. S2 (`dc88d6d`) corrected this count and populated
all 11 entries. Previously, `guard let expectedHash = file.sha256 else { continue }`
inside `AcervoManager.withComponentAccess` short-circuited for every file.

**Work**:
1. Pull the canonical MLX artifacts for each component from the CDN (same slugs used in
   `ensure-model-cdn.yml`):
   - `intrusive-memory_t5-xxl-int4-mlx/{config.json, tokenizer.json, tokenizer_config.json, model.safetensors}`
   - `intrusive-memory_sdxl-vae-fp16-mlx/{config.json, model.safetensors}`
2. Compute SHA-256 via `shasum -a 256 <file>`; record `expectedSizeBytes` via `stat -f%z`.
3. Rewrite each `ComponentFile` to pass both: `ComponentFile(relativePath: …, expectedSizeBytes: …, sha256: "…")`.
4. Extend `CatalogRegistrationTests` with a suite that asserts every `ComponentFile.sha256`
   is non-nil, every `expectedSizeBytes` is > 0, and digest hex strings are 64 lowercase chars.

**Exit**: Running `xcodebuild test -scheme SwiftTuberia-Package …` passes; an
integration test (REQ-INT-01 below) observes `AcervoError.integrityCheckFailed` when a
file is corrupted on disk.

---

### REQ-T5: Bump `SwiftAcervo` Minimum Floor in `Package.swift`

**Priority**: 🔴 CRITICAL — consumers resolving to SwiftAcervo 0.5.x will not have
`withComponentAccess`/`ComponentHandle` symbols and will fail to build.

**Files**:
- `Package.swift:23`

**Current state**: `.package(url: "https://github.com/intrusive-memory/SwiftAcervo.git", from: "0.5.6")`.
`Package.resolved` is at `0.7.2`, but `from:` governs fresh resolution.

**Work**:
1. Change the `from:` bound to the smallest SwiftAcervo version that ships the v2 API
   (`withComponentAccess`, `ComponentFile.sha256`, `ComponentHandle`, `Acervo.register`,
   `Acervo.ensureComponentReady`, `AcervoError.integrityCheckFailed`). Inspection of
   `/Users/stovak/Projects/SwiftAcervo/API_REFERENCE.md` confirms these are present at
   0.7.x; use `from: "0.7.2"` to match the currently resolved revision.
2. Run `xcodebuild -resolvePackageDependencies` and commit updated `Package.resolved`.
3. Reconcile AGENTS.md (v0.3.0 entry says "Bumped SwiftAcervo to 0.6.0") — either update
   the release note to the actual bump (0.7.2) or add a new entry documenting the
   additional floor bump.

**Exit**: Fresh `swift package reset && xcodebuild build` resolves ≥ 0.7.2; no source
changes required; all tests still pass.

---

### REQ-PIPE-01: Call `Acervo.ensureComponentReady(_:)` Before Weight Loading

**Priority**: 🔴 CRITICAL — without this, `DiffusionPipeline.loadModels` throws
`componentNotDownloaded` on first use and never auto-downloads, violating both
`requirements/PROTOCOLS.md:16` and `architecture/PROTOCOLS.md:40`.

**Files**:
- `Sources/Tuberia/Pipeline/DiffusionPipeline.swift:173-217`

**Work**:
1. At the start of the `for segment…` loop in `loadModels(progress:)`, for each weighted
   segment that has a non-nil `componentId`, call
   `try await Acervo.ensureComponentReady(componentId, progress: …)` **before**
   invoking `WeightLoader.load`.
2. Thread the existing `(Double, String)` progress callback through the Acervo progress
   closure so the download phase emits pre-load progress instead of a hung bar.
3. Add a `downloadProgress` enum case (or a new `PipelineProgress.downloading(component:fraction:)`)
   if finer-grained reporting is wanted — otherwise fold into the existing `progress(loadedCount/totalSegments, componentName)` tick stream.
4. Confirm behavior against both pathways in tests: component not on disk (downloads),
   and component already cached (no network).

**Exit**: Calling `generate()` on a clean cache downloads required components
automatically. PROTOCOLS.md step sequencing is satisfied.

---

### REQ-PIPE-02: Wire `MemoryManager.hardValidate()` Into the Load Path

**Priority**: 🔵 HIGH — documented contract gap. Large models currently OOM without
the documented soft/hard guard.

**Files**:
- `Sources/Tuberia/Pipeline/DiffusionPipeline.swift:173-217`
- `Sources/Tuberia/Infrastructure/MemoryManager.swift:81-99`
- `requirements/INFRASTRUCTURE.md:154-218`

**Work**:
1. At entry to `loadModels(progress:)`, compute the peak-memory requirement from
   `_memoryRequirement.peakMemoryBytes` and call
   `await MemoryManager.shared.hardValidate(requiredBytes: peak)`.
2. On `PipelineError.insufficientMemory` (or whatever `hardValidate` throws — currently
   generic `MemoryError`; see INFRASTRUCTURE.md canonical definition), surface a
   pipeline-specific error variant so the caller can trigger a phased reload.
3. Alternatively, if phased loading is the intended path, call
   `softCheck(requiredBytes: peak)`; on `false`, switch to per-phase `hardValidate`
   before each phase (encoder first, then backbone+decoder).
4. Document the chosen strategy inline and update `requirements/INFRASTRUCTURE.md` if
   the design changed.

**Exit**: A unit test that stubs `MemoryManager.availableMemory` below the pipeline's
`peakMemoryBytes` observes a thrown `PipelineError` from `loadModels`.

---

### REQ-PIPE-03: Replace Positional `findComponentId(for:)` With Recipe Role Map

**Priority**: 🔵 HIGH — latent bug. Any recipe that adds scheduler/renderer/auxiliary
component IDs (e.g. a scheduler with learned weights, a tokenizer-only component) silently
mis-associates IDs with segments.

**Files**:
- `Sources/Tuberia/Pipeline/DiffusionPipeline.swift:515-530`
- `Sources/Tuberia/Pipeline/PipelineRecipe.swift`

**Current state**: `findComponentId(for:)` indexes `_allComponentIds` by a hard-coded
`[0]=encoder, [1]=backbone, [2]=decoder` convention.

**Work**:
1. Add `var componentIdFor: [PipelineRole: String] { get }` (or equivalent) to
   `PipelineRecipe`. Every existing recipe either gains a generated default (derived from
   `allComponentIds` + `PipelineRole` iteration order, preserving current behavior) or an
   explicit mapping.
2. Replace `_allComponentIds: [String]` with `_componentIdByRole: [PipelineRole: String]`
   on `DiffusionPipeline`.
3. Rewrite `findComponentId(for:)` as a dictionary lookup. Delete the positional fallback.
4. Add a `PipelineAssemblyTests` case that builds a mock recipe whose role-id pairs are
   reversed and confirms each segment loads its own weights (not its neighbor's).

**Exit**: No positional assumptions remain. A test that reverses recipe ID order still
passes.

---

### REQ-INT-01: End-to-End `withComponentAccess` Integration Tests

**Priority**: 🔵 HIGH — current coverage validates registration metadata only; no test
exercises the integrity-verified read path or its failure modes.

**Files (new)**:
- `Tests/TuberiaCatalogTests/WeightLoaderIntegrationTests.swift`
- `Tests/TuberiaCatalogTests/ComponentIntegrityTests.swift`

**Work**:
1. **Happy path**: stage a temporary App Group directory with a synthetic safetensors
   file whose SHA-256 matches a test-only `ComponentDescriptor`. Invoke
   `WeightLoader.load(componentId:…)` through `AcervoManager.withComponentAccess` with a
   test base directory (use the internal `withComponentAccess(_:in:perform:)` overload).
   Assert `ModuleParameters` contains the expected key.
2. **Integrity failure**: stage the same directory, but corrupt a byte in the safetensors
   file. Assert the load throws `AcervoError.integrityCheckFailed` (mapped to
   `PipelineError.weightLoadingFailed`).
3. **Not-downloaded failure**: register a descriptor but stage no files. Assert
   `PipelineError.modelNotDownloaded`.
4. **LoRA `withLocalAccess`**: stage a bare safetensors file on disk, call
   `LoRALoader.loadAdapterWeights(config: .init(localPath: url.path, …))`, assert
   non-empty params.
5. All new tests honor `TESTING.md` rules (no real weights, no network, no timed waits).
   Use small synthetic tensors.

**Exit**: New tests pass on CI; deliberately corrupting a descriptor byte in the staged
file causes a clean, observable failure (not a silent mis-read).

---

### REQ-CDN-01: Verify Uploaded Manifest Matches Repository Descriptors

**Priority**: 🟡 MEDIUM — guards against checksum drift once REQ-T4 lands.

**Files**:
- `.github/workflows/ensure-model-cdn.yml`
- (possibly new) `Tools/verify_component_manifest.swift`

**Work**:
1. After the `Verify upload` step, add a `Verify manifest matches source` step that:
   - Downloads the uploaded `manifest.json`
   - Parses each `{path, size, sha256}` entry
   - Loads the corresponding `ComponentDescriptor` from
     `CatalogRegistration.swift` (either via a small helper executable target or by
     running a `swift test` that dumps descriptors to JSON).
   - Fails the job if any per-file `sha256` or `expectedSizeBytes` in the descriptor
     differs from the manifest.
2. Treat a mismatch as a hard failure with a clear message — the fix is to update
   `CatalogRegistration.swift` and commit.

**Exit**: A deliberate mismatch (edit one hex digit in CatalogRegistration.swift) causes
the next workflow run to fail with a diff-style error; reverting the edit passes.

---

### REQ-DOC-01: Update AGENTS.md / Architecture Docs for v2 Completion

**Priority**: 🟡 MEDIUM — documentation consistency.

**Files**:
- `AGENTS.md` (Recent Changes section)
- `architecture/PROTOCOLS.md`, `architecture/INFRASTRUCTURE.md`
- `requirements/INFRASTRUCTURE.md` ("Audit Checklist" section already claims items are
  complete — re-verify each one post-T4/T5)

**Work**:
1. Add a v0.3.7 (or next) changelog entry summarizing: minimum Acervo bump, per-file
   checksums added, ensureComponentReady in load path, hardValidate wired in, positional
   findComponentId replaced.
2. Update `requirements/INFRASTRUCTURE.md:209-218` "Audit Checklist" to reflect the true
   state of each item after the above sorties land (several currently marked `[x]` that
   are not actually implemented — e.g. `MemoryManager.hardValidate()` called before
   loading).
3. Cross-link this REQUIREMENTS.md from AGENTS.md's top-level "Architecture" section.

**Exit**: A fresh contributor reading AGENTS.md + this file can reconstruct the v2
integration state without running `grep`.

---

## Non-Blocking / Deferred Items

Captured from `AUDIT_FINDINGS.md` "Non-Blocking Items" plus new observations:

- **LoRA descriptor checksums** — when LoRAs are ever shipped via CDN (not just local
  paths), they need the same SHA-256 treatment. Today LoRA entries use `componentId` only
  as an opaque pass-through to `WeightLoader.load`, so REQ-T4's pattern applies unchanged
  — but no catalog LoRAs exist yet.
- **Model-plugin backbones** (pixart-swift-mlx, flux-2-swift-mlx) register their own
  `ComponentDescriptor`s. They must apply REQ-T4 independently in their own repos; this
  REQUIREMENTS.md does not govern them.
- **`CLIPEncoder`, `DDPMScheduler`** — declared in `requirements/CATALOG.md` but not yet
  implemented. Out of scope for v2 integration; create their descriptors when those
  components land.
- **`Renderer` components** (`ImageRenderer`, `AudioRenderer`) — weightless, need no
  Acervo descriptor. Confirm no regression introduces one.

---

## Cross-References

- **Master mission index**: `/Users/stovak/Projects/REQUIREMENTS.md` (Work Unit 2,
  Sorties 2.1–2.3)
- **Audit source**: `AUDIT_FINDINGS.md` (Sorties T1–T5) — upon completion of REQ-T4 and
  REQ-T5, mark the T4/T5 exit criteria checked.
- **Upstream SwiftAcervo API**: `/Users/stovak/Projects/SwiftAcervo/API_REFERENCE.md`
- **Architectural spec**: `docs/incomplete/REQUIREMENTS_ARCHITECTURE.md`,
  `requirements/{PROTOCOLS,PIPELINE,CATALOG,INFRASTRUCTURE,TESTING,TEST_COVERAGE_GAPS,INFERENCE}.md`,
  `architecture/{PROTOCOLS,PIPELINE,CATALOG,INFRASTRUCTURE}.md`

---

## Definition of Done (Master Acceptance)

SwiftTuberia's SwiftAcervo v2 integration is **complete** when:

- [ ] All rows in the Status Snapshot read ✅ DONE.
- [ ] `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'`
      passes on a clean clone.
- [ ] `xcodebuild test …` with a locally corrupted component file surfaces
      `AcervoError.integrityCheckFailed` (propagated as `PipelineError.weightLoadingFailed`).
- [ ] Fresh `swift package reset && xcodebuild build` resolves SwiftAcervo ≥ 0.7.2.
- [ ] `ensure-model-cdn.yml` workflow's manifest-verification step fails on a deliberate
      checksum mismatch.
- [ ] Master index `/Users/stovak/Projects/REQUIREMENTS.md` SwiftTuberia row moves from
      🔴 RED 0% to 🟢 GREEN 100%, and Sorties 2.1–2.3 + T1–T5 all marked 🟢 COMPLETE.

---

## History

| Date | Event |
|------|-------|
| 2026-04-18 | `AUDIT_FINDINGS.md` created with Sorties T1–T5. |
| 2026-04-20 | This file created. T1–T3 verified complete; T4 (checksums), T5 (dependency floor), and three new sorties (REQ-PIPE-01/02/03, REQ-INT-01, REQ-CDN-01, REQ-DOC-01) identified as outstanding. |
