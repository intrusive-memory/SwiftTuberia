# SwiftTuberia — SwiftAcervo 0.16.0 migration TODO

Generated: 2026-05-23
Source of truth: `/Users/stovak/Projects/SwiftAcervo/UPGRADING.md`
Current pin: SwiftAcervo 0.14.0 (`Package.swift:64`, `Package.resolved` revision `15fd376`)
Target pin: SwiftAcervo 0.16.0

## Executive summary

SwiftTuberia uses **only the v2 component-keyed Acervo API**
(`Acervo.register`, `Acervo.component`, `Acervo.ensureComponentReady`,
`AcervoManager.withComponentAccess`, `AcervoManager.withLocalAccess`,
`AcervoError.componentNot*`, `AcervoDownloadProgress`,
`SwiftAcervo.ComponentDescriptor`). Every one of these survives 0.16.0
unchanged.

SwiftTuberia has **zero** call sites for `Acervo.availability(_:)` or
any `switch` over `ModelAvailability`, so the headline 0.16.0 break
(new `.partial` case requiring switch-exhaustiveness updates) does not
apply.

Therefore the migration is dominated by **two ops tasks**:

1. Bump the SwiftAcervo version pin in `Package.swift`.
2. Re-ship the two CDN-hosted models so their manifests carry the now-
   required `primaryRepo` / `components` wire fields (see
   `../MODELS-TO-SHIP.md`).

There are zero required source edits in `Sources/`. There are zero
required test-source edits in `Tests/`. Everything below is verification
and (optional) hygiene.

---

## Group A — Version pin (required, do first)

### A1. `Package.swift:59-65` — bump SwiftAcervo dependency to 0.16.0

Current:

```swift
sibling(
  "SwiftAcervo",
  remote: "https://github.com/intrusive-memory/SwiftAcervo.git",
  from: "0.14.0"),
```

Replacement:

```swift
sibling(
  "SwiftAcervo",
  remote: "https://github.com/intrusive-memory/SwiftAcervo.git",
  from: "0.16.0"),
```

Caveats:
- `sibling(...)` resolves to `../SwiftAcervo` when present locally and to
  the remote pin otherwise (see `Package.swift:23-29`). For local
  development, confirm `../SwiftAcervo` is on a commit at or after
  `v0.16.0`.
- The `from:` form translates to `.upToNextMajor(from: "0.16.0")`. Since
  0.x releases are not yet 1.x, SemVer treats every minor as a major
  break — so `.upToNextMajor(from: "0.16.0")` will accept `0.16.x` only.
  This matches what we want here.
- After bumping, regenerate `Package.resolved` (allowed during the
  migration; not a source edit per se). The new resolved revision for
  `SwiftAcervo` should be the `v0.16.0` tag SHA.

### A2. `Package.resolved` — refresh after A1

Not edited by hand. Run `swift package resolve` (or open in Xcode) to
let SwiftPM rewrite the SwiftAcervo pin block. Verify the new entry
shows `"version" : "0.16.0"`.

---

## Group B — Source call-site audit (zero edits required — verify only)

### B1. `Sources/Tuberia/Pipeline/ComponentReadinessService.swift` — no change

The protocol surface uses `AcervoDownloadProgress` and the production
conformer calls `Acervo.ensureComponentReady(_:progress:)`. Both
survive 0.16.0 unchanged. After A1, verify the file compiles untouched.

### B2. `Sources/Tuberia/Pipeline/DiffusionPipeline.swift:602` — no change

Calls `componentReadinessService.ensureComponentReady(componentId) {
downloadProgress in ... }`. Goes through the protocol seam; the
underlying Acervo call is `Acervo.ensureComponentReady(_:progress:)`
which is unchanged.

### B3. `Sources/Tuberia/Infrastructure/WeightLoader.swift` — no change

Uses `AcervoManager.shared.withComponentAccess` (line 45),
`AcervoManager.shared.withLocalAccess` (line 194), `handle.urls(matching:
".safetensors")` (lines 53, 195), and catches the v2 component errors:
`componentNotDownloaded`, `componentNotRegistered`, `componentNotHydrated`,
`integrityCheckFailed` (lines 113-146 and 215-226). All preserved in
0.16.0 — confirmed against
`/Users/stovak/Projects/SwiftAcervo/Sources/SwiftAcervo/AcervoManager.swift`,
`ComponentHandle.swift`, and `AcervoError.swift`.

### B4. `Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` — no change

Uses `SwiftAcervo.ComponentDescriptor(...)` (lines 11, 29), `Acervo.register([...])`
(line 47), `Acervo.component(componentId)` (lines 87, 92), and
`Acervo.ensureComponentReady(componentId)` (line 105). All v2 component-
catalog APIs that survive 0.16.0 (see
`/Users/stovak/Projects/SwiftAcervo/Sources/SwiftAcervo/Acervo+ComponentCatalog.swift`
and `Acervo+ComponentRegistration.swift`).

The `@available(*, deprecated, ...)` annotation on
`CatalogRegistration.ensureComponentReady(_:)` (lines 99-106) is purely
SwiftTuberia-internal and unrelated to the 0.16 migration. Leave it in
place.

### B5. `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift:59-63` — no change

Calls `AcervoManager.shared.withComponentAccess(...) { handle in
handle.rootDirectoryURL }`. `rootDirectoryURL` on `ComponentHandle` is
preserved in 0.16.0 — confirmed in
`/Users/stovak/Projects/SwiftAcervo/Sources/SwiftAcervo/ComponentHandle.swift`.

### B6. `Sources/TuberiaCatalog/Encoders/T5XXLEncoderConfiguration.swift` — no change

Holds `componentId: String` only. Not affected.

### B7. `Sources/TuberiaCatalog/Decoders/SDXLVAEDecoderConfiguration.swift` — no change

Holds `componentId: String` only. Not affected.

### B8. `Sources/Tuberia/Pipeline/LoRAConfig.swift` — no change

Holds optional `componentId` / `localPath` strings; never resolves them
itself.

---

## Group C — Test call-site audit (zero edits required — verify only)

### C1. `Tests/TuberiaTests/Support/TestEnvironment.swift` — no change

`setenv("ACERVO_APP_GROUP_ID", ...)` bootstrap. App Group resolution is
unchanged in 0.16.0.

### C2. `Tests/TuberiaCatalogTests/Support/TestEnvironment.swift` — no change

Same as C1.

### C3. `Tests/TuberiaGPUTests/Support/TestEnvironment.swift` — no change

Same as C1.

### C4. `Tests/TuberiaGPUTests/ContractTests/PipelineLoadModelsTests.swift:33-52` — no change

The `ComponentReadinessSpy` conforms to `ComponentReadinessService` and
synthesizes `AcervoDownloadProgress(fileName:bytesDownloaded:totalBytes:
fileIndex:totalFiles:)`. Both the protocol and the `AcervoDownloadProgress`
memberwise init are unchanged in 0.16.0.

### C5. `Tests/TuberiaTests/RecipeRoleMapTests.swift:172-183` — no change

Same pattern as C4; spy conformer only. No `ModelAvailability` switch.

### C6. `Tests/TuberiaCatalogTests/CatalogRegistrationTests.swift` — no change

Asserts against `descriptor?.id`, `descriptor?.type`, `descriptor?.repoId`,
`descriptor?.estimatedSizeBytes == 0`, `descriptor?.isHydrated == false`,
`descriptor?.files.count == 0`, `descriptor?.metadata[...]`. Every accessor
on `ComponentDescriptor` referenced here is preserved in 0.16.0
(verified against
`/Users/stovak/Projects/SwiftAcervo/Sources/SwiftAcervo/ComponentRegistry.swift`).

### C7. `Tests/TuberiaCatalogTests/WeightLoaderIntegrationTests.swift` — no change

Exercises `LoRALoader.loadAdapterWeights(...)` over a local
`adapter.safetensors`. Goes through `AcervoManager.withLocalAccess` —
unchanged in 0.16.0.

### C8. `Tests/TuberiaTests/MemoryGuardTests.swift`, `Tests/TuberiaTests/TuberiaTelemetryLoRATests.swift` — no change

Only `import SwiftAcervo` + `TestEnvironment.ensureAcervoAppGroup()`.
Nothing that breaks under 0.16.0.

---

## Group D — Things 0.16.0 explicitly flagged that DO NOT apply here

These are listed so a future agent does not waste time chasing them:

- **`ModelAvailability.partial` switch exhaustiveness.** No `switch` in
  the SwiftTuberia tree binds `ModelAvailability`. Grep verified:
  `rg "ModelAvailability|case \.partial|case \.notAvailable" Sources/ Tests/`
  returns nothing.
- **Slug-keyed `availability(slug:url:)` / `ensureAvailable(slug:url:)` /
  `deleteModel(slug:url:)`.** SwiftTuberia addresses every model by
  component ID through the v2 catalog API, never by deployment slug.
  No call sites to migrate.
- **`CDNManifest` test fixtures with `primaryRepo` / `components`.**
  SwiftTuberia never hand-writes a `CDNManifest` JSON or in-memory
  fixture. The 0.16.0 strict-decode requirement is invisible to this
  consumer.
- **`Acervo.listModels()` validity filtering.** Not called.
- **`Acervo.gcEmptyModelDirectories()`.** Not used; nothing to adopt.
- **`acervo ship --slug`, `--spec`, `--dry-run`, `--output-dir`.** CLI
  flags — they belong to the model-shipping step (see MODELS-TO-SHIP.md
  in the parent directory), not to SwiftTuberia itself.
- **Source decomposition of `Acervo.swift` into 15 `Acervo+*.swift`
  files.** No SwiftTuberia tool, doc, or AGENTS.md note pins a line
  range inside SwiftAcervo source. Grep verified:
  `rg "Acervo\.swift:[0-9]" docs/ AGENTS.md CLAUDE.md GEMINI.md`
  returns nothing.
- **0.15.0 `aws` CLI removal.** SwiftTuberia CI never installed `aws`
  for SwiftAcervo. Nothing to delete.
- **0.15.0 default orphan prune in `acervo ship`.** Operator-side
  concern handled in MODELS-TO-SHIP.md, not here.

---

## Group E — Post-bump verification protocol

Run after A1 + A2, before merging:

1. `swift package resolve` — confirm `Package.resolved` shows SwiftAcervo
   at 0.16.0 (and no transitive surprises).
2. `make test` (or the project's standard test target) — every test must
   pass without source edits. If anything fails to compile, the audit
   above missed a call site; do NOT ad-hoc patch it, return to this
   TODO and amend.
3. Spot-check telemetry: the GLASS PIPES per-step events
   (`backboneForwardComplete`, `denoiseStepComplete`,
   `componentReadinessChecked`, `weightLoadStart`, `weightLoadComplete`)
   must still emit with the same field names. They are SwiftTuberia-
   owned and not affected by SwiftAcervo, but the verification belongs
   here because the regression surface touches `DiffusionPipeline`'s
   download-progress folding (line 602).

## Group G — CI test-shipping dependency audit (2026-05-23)

Explicit verification that no test in this repo depends on
`t5-xxl-encoder-int4` or `sdxl-vae-decoder-fp16` being shipped to the
CDN, so CI cannot fail in the gap between pin-bump and re-ship.

Files grepped: every `Tests/**/*.swift` for `ensureComponentReady`,
`withComponentAccess`, `t5-xxl-encoder-int4`, `sdxl-vae-decoder-fp16`,
`T5XXLEncoder`, `SDXLVAEDecoder`, `loadTokenizer`.

Pattern observed per test file:

- `Tests/TuberiaCatalogGPUTests/T5XXLEncoderTests.swift` — constructs
  `T5XXLEncoder` directly with `T5XXLEncoderConfiguration` and tests
  shape contracts / key mapping. No download.
- `Tests/TuberiaCatalogGPUTests/SDXLVAEDecoderTests.swift` — same
  pattern: direct decoder construction, shape contracts. No download.
- `Tests/TuberiaCatalogTests/T5KeyMappingTests.swift` — pure static
  key-mapping table tests via `T5XXLEncoder.mapKey(_:)`. No I/O.
- `Tests/TuberiaCatalogTests/SDXLVAEDecoderTests.swift` — synthetic
  `ModuleParameters` via `apply(weights:)`. No download.
- `Tests/TuberiaCatalogTests/SDXLVAEModelTests.swift` — internal
  `SDXLVAEDecoderModel` direct forward-pass with zero-init weights.
  No download.
- `Tests/TuberiaCatalogTests/T5TokenizerIntegrationTests.swift` —
  uses placeholder tokenization for the main suites; the
  `T5LoadTokenizerPresenceTests` suite explicitly passes
  `componentId: "nonexistent-test-component"` and asserts that
  `loadTokenizer()` silently handles a missing Acervo component.
- `Tests/TuberiaCatalogTests/CatalogRegistrationTests.swift` — only
  asserts that descriptors exist with the right metadata; never
  triggers a download or file access.
- `Tests/TuberiaCatalogTests/WeightLoaderIntegrationTests.swift` —
  uses `AcervoManager.withLocalAccess` against on-disk test fixtures
  written by the test itself; the t5/vae component slugs are not
  referenced.
- `Tests/TuberiaGPUTests/WeightLoaderTests.swift`,
  `Tests/TuberiaGPUTests/LoRATests.swift` — likewise: zero hits for
  `ensureComponentReady`, `withComponentAccess`, or the two component
  IDs. Confirmed by grep.
- `Tests/TuberiaGPUTests/ContractTests/PipelineLoadModelsTests.swift`
  and `Tests/TuberiaTests/RecipeRoleMapTests.swift` — use the
  `ComponentReadinessSpy` (a protocol-seam test double) and never
  reach the real `Acervo.ensureComponentReady` path.

Conclusion: nothing to skip, gate, or annotate. CI will stay green
during the gap between bumping the SwiftAcervo pin to 0.16.0 and
re-shipping the two CDN models. The model re-ship is required for
*end-user runtime correctness* (fresh strict-decode manifest fetches),
not for CI test health.

If a future contributor adds a test that *does* call
`Acervo.ensureComponentReady("t5-xxl-encoder-int4")` or
`ensureComponentReady("sdxl-vae-decoder-fp16")` without a spy, that
test MUST be gated behind an env-var check (e.g.
`ProcessInfo.processInfo.environment["TUBERIA_RUN_LIVE_ACERVO"] != nil`)
or marked `.disabled` so CI does not regress.

## Group F — Optional hygiene (not blocking)

Pure documentation polish; not required for the upgrade to land.

- `Sources/Tuberia/Infrastructure/WeightLoader.swift:10` — comment
  references "Acervo v2 `withComponentAccess`". v2 nomenclature still
  applies in 0.16.0 (the component API has not been renamed), so the
  comment is technically accurate. Leave as-is unless project style
  prefers stripping version qualifiers from doc comments.
- `Tests/TuberiaTests/Support/TestEnvironment.swift:5` — comment cites
  "SwiftAcervo 0.10+". Could be updated to "SwiftAcervo 0.10+ (still
  applies through 0.16)". Not required.
