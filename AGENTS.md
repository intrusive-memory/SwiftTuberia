# AGENTS.md

This file provides comprehensive documentation for AI agents working with the SwiftTuberia codebase.

**Version**: 0.7.2-dev

---

## Recent Changes

### v0.7.0 — Telemetry instrumentation (GLASS PIPES) + SwiftAcervo floor bump

Minor release adding a structured telemetry surface across the diffusion pipeline plus a SwiftAcervo dependency bump.

- **Telemetry events** (`TuberiaTelemetryEvent`, `TuberiaTelemetryReporter`, `TuberiaTensorStat`): emit lifecycle, assembly, memory, weight, LoRA, text-encoder, scheduler, denoise-loop, CFG-cast, anomaly, backbone/decoder/renderer events from the diffusion pipeline. Reporters are injectable via defaulted parameters; passing nothing is a no-op.
- **Five instrumented protocol seams**: assembly, denoise loop, CFG cast, anomaly detection, LoRA loading. See `Sources/Tuberia/Pipeline/DiffusionPipeline+Telemetry.swift`.
- **SwiftAcervo floor bump**: `0.11.1` → `0.13.0`.
- **swift-tokenizers pin**: held at `0.5.x` (upstream `0.6.x` ships broken FFI).

### v0.6.5 — swift-tokenizers 0.5.0 floor bump (broken-resolution fix)

Patch release that unbreaks SPM resolution against upstream swift-tokenizers.

- **Floor bump**: `swift-tokenizers` `0.4.3` → `0.5.0` (latest published release — retains `.upToNextMajor`).
- **Trait declaration removed**: previous releases passed `traits: ["Swift"]` to swift-tokenizers. As of 0.5.0, the package dropped its trait system and is Rust-backend-only — passing the trait now fails resolution. The trait argument is gone from `Package.swift`.
- **Consumer impact**: SwiftTuberia consumers will now resolve the binary `TokenizersRust` XCFramework that swift-tokenizers 0.5.0 declares as a binary target. Min platforms (iOS 17 / macOS 14) are well below SwiftTuberia's iOS 26 / macOS 26 floor; no platform changes required.
- **No source changes**: `AutoTokenizer.from(directory:)` and the `Tokenizer` protocol surface used by `T5XXLEncoder` are unchanged across the 0.4 → 0.5 boundary.

### v0.6.4 — Maintenance release

Patch release with no functional changes. Refreshes the development snapshot to a clean release tag and updates documentation/CI hygiene.

- Documentation version refs synchronized to v0.6.4.
- CI workflow actions audited.

### v0.6.3 — SwiftAcervo v0.11.1 floor bump + workflow env exports

Patch release aligned with OPERATION GROUPHOUSE MUSTER canary validation.

- **Dependency floor bump**: `SwiftAcervo` `0.10.0` → `0.11.1` (latest published release — retains `.upToNextMajor`).
- **CI workflow env exports**: `ACERVO_APP_GROUP_ID` now exported at the workflow job level (already merged via Sortie 1.7 / PR #30), ensuring test runners never silently trap on missing App Group configuration.
- **Documentation**: AGENTS.md updated with current `ACERVO_APP_GROUP_ID` env var contract (already merged via Sortie 2.7 / PR #31).

### v0.6.2 — SPM checkouts sibling helper fix

Patches the `sibling()` helper in `Package.swift` so it no longer false-positives when SwiftTuberia is consumed as a transitive dependency.

The previous helper preferred a `.package(path: "../<name>")` reference whenever `../<name>` existed on disk — intended for local dev workspaces. Inside Xcode's `DerivedData/.../SourcePackages/checkouts/` (or SwiftPM's `.build/checkouts/`) every sibling package lives in the same directory, so the existence check succeeded for every dependency and the helper switched them all to unversioned path references. Those collided with the same identities pulled in via remote URLs elsewhere in the graph and aborted resolution with `Conflicting identity for swiftacervo`.

Detect the checkout context by inspecting `#filePath` for the well-known SPM path segments and skip the sibling shortcut there. Local dev workflow and CI behavior are unchanged.

- Dependency floor bump: `SwiftAcervo` `0.8.4` → `0.8.5` (latest published patch, retains `.upToNextMajor`).

### v0.6.1 — Fix T5RMSNorm fp16 sum overflow

Casts the variance reduction in `T5RMSNorm` to fp32 before `mean(axis: -1)`, matching diffusers' `T5LayerNorm`. Without the cast, T5-XXL token embeddings — whose channel magnitudes reach ~100 — overflow the fp16 intermediate sum at hidden_dim=4096: any per-token variance > ~16 pushes the running sum past fp16's finite range (~65504), the sum rounds to `+inf`, and `rsqrt(+inf) = 0` collapses the entire RMSNorm output to literally zero for that token. The corruption is silent and propagates through every downstream layer.

- **Impact on PixArt canonical fixture prompt**: 10 of 13 real (non-padding) tokens were hitting the overflow. Output mean cosine vs. the diffusers reference was 0.36 — every content token diverged, only EOS matched. With the fp32 cast, mean cosine is ≥0.988 at every layer and end-to-end PixArt generation produces a recognizable photographic image instead of a flat cartoon abstraction.
- **Residual drift, documented not fixed**: even with the fp32 variance cast, post-final-norm cosine measures 0.988 rather than 1.000 across 24 transformer blocks. This is fp16 accumulation noise in the rest of the stack, not a correctness bug. A `TODO` in `T5TransformerEncoder` notes that bit-perfect parity with diffusers would require running the full block stack in fp32. Out of scope for this patch.
- Dependency floor bumps: `SwiftAcervo` `0.8.3` → `0.8.4`; `swift-tokenizers` `0.4.2` → `0.4.3` (latest published patch releases — both retain `.upToNextMajor`).

### v0.6.0 — OPERATION VANISHING MANIFEST: SwiftAcervo v2 compliance

Aligns SwiftTuberia with the SwiftAcervo v2 contract — no manifest synthesis, typed error handling, and the `rootDirectoryURL` access pattern.

- **Bare `ComponentDescriptor` declarations** (Tuberia-S1) — `CatalogRegistration` no longer carries per-file SHA-256 / size shadow data. SwiftAcervo's CDN manifest is the sole source of truth; `ComponentFile` entries are reduced to `path` only.
- **Deprecated `CatalogRegistration.ensureComponentReady`** (Tuberia-S1) — call sites should use `Acervo.ensureComponentReady` (with progress) or `ComponentReadinessService` directly. Shim retained with `@available(*, deprecated)` for one release.
- **Typed `AcervoError` handling in `WeightLoader`** (Tuberia-S2) — `loadComponent` and `loadFromPath` catch `AcervoError.integrityCheckFailed(file:expected:actual:)` explicitly and rethrow as `PipelineError.weightLoadingFailed`. Dead `validateChecksum` / `validateSize` paths removed.
- **`T5XXLEncoder` uses `rootDirectoryURL`** (Tuberia-S3) — replaces direct path manipulation with the v2 access pattern. Adds precondition docstring on directory shape.
- **`Package.resolved` untracked** — restored to `.gitignore`-honoured state. Library convention is to let consumers resolve.
- **Dependency floors tightened**:
  - `SwiftAcervo`: `.upToNextMajor(from: "0.7.3")` → `.upToNextMajor(from: "0.8.3")` — `rootDirectoryURL` (consumed by `T5XXLEncoder.loadTokenizer`) was added in 0.8.3.
  - `swift-tokenizers`: `.upToNextMajor(from: "0.3.2")` → `.upToNextMajor(from: "0.4.2")` — pin to current latest published release.

### v0.5.0 — Tokenizer dependency migration (swift-transformers → swift-tokenizers)

Mirrors `mlx-audio-swift`'s tokenizer migration. `Package.swift` dependencies:
- Removed: `huggingface/swift-transformers` (`Transformers` product)
- Added: `DePasqualeOrg/swift-tokenizers` `.upToNextMajor(from: "0.3.2")` with `traits: ["Swift"]` (`Tokenizers` product)
- `SwiftAcervo`: `.upToNextMajor(from: "0.7.2")` → `.upToNextMajor(from: "0.7.3")`

`TuberiaCatalog` now consumes only the `Tokenizers` product — the full `Transformers` product is no longer needed, trimming the transitive dep graph (Generation, Hub, Agents, etc.). `T5XXLEncoder` migrated to the new API: `AutoTokenizer.from(modelFolder:)` → `AutoTokenizer.from(directory:)` (one call site). No public API changes for consumers.

### v0.4.0 — OPERATION RIVETED PIPEWORK release + dependency floor bumps

Bundles v0.3.7/v0.3.8/v0.3.9 work (none of which shipped independently) plus an MLX test parallelism fix into a single minor release. `Package.swift` dependency floors tightened:
- `mlx-swift`: `from: "0.30.2"` → `.upToNextMajor(from: "0.31.3")`
- `SwiftAcervo`: `from: "0.7.2"` → `.upToNextMajor(from: "0.7.2")`
- `swift-transformers`: `from: "1.1.6"` → `.upToNextMajor(from: "1.3.0")`

**MLX test parallelism (`86b2b2d`)** — `make test` and `.github/workflows/tests.yml` now pass `-parallel-testing-enabled NO`. MLX owns a process-global Metal GPU command stream; Swift Testing's default cross-suite parallelism races on the shared command buffer, tripping `-[_MTLCommandBuffer addCompletedHandler:] 'Completed handler provided after commit call'` and aborting with SIGABRT. Suite-level `.serialized` traits only serialize within a suite, not across sibling suites. Any ad-hoc `xcodebuild test` invocation must set the flag too.

See v0.3.9, v0.3.8, and v0.3.7 entries below for the pipeline-integrity work rolled into this release.

### v0.3.9 — Delegate File Integrity to SwiftAcervo CDN Manifest (Correction)

Tuberia no longer stores per-file SHA-256 checksums or `expectedSizeBytes` values in `CatalogRegistration.swift`; SwiftAcervo's CDN manifest download and per-file verification path is the single source of truth for all 11 `ComponentFile` entries (REQ-T4). The `VerifyComponentManifest` SwiftPM executable tool and its corresponding "Verify manifest matches source" CI step have been removed from `Package.swift` and `.github/workflows/ensure-model-cdn.yml` respectively; SwiftAcervo's built-in integrity mechanism (`AcervoError.integrityCheckFailed`, `manifestIntegrityFailed`) is authoritative and needs no caller-side double (REQ-CDN-01). This supersedes the v0.3.8 language for REQ-T4 and REQ-CDN-01 — both were architecturally mis-implemented by S2 (`dc88d6d`) and S7 (`f4e6939`).

### v0.3.8 — SwiftAcervo v2 Integration Complete (OPERATION RIVETED PIPEWORK)

- **SwiftAcervo floor bumped to 0.7.2** (REQ-T5, S1 `0aa8fcf`) — `Package.swift` now declares `from: "0.7.2"`, ensuring fresh resolutions never regress to v1 symbols. Resolves the gap between `Package.resolved` pin and declared floor.
- **SHA-256 checksums on all 11 ComponentFile descriptors** (REQ-T4, S2 `dc88d6d`) — all `ComponentFile` entries in `CatalogRegistration.swift` now carry `sha256:` and `expectedSizeBytes:`. Coverage: 5 T5-XXL safetensors shards + 4 T5-XXL metadata files + 2 SDXL VAE files = 11 total (plan originally assumed 6; T5 is sharded). Integrity-verification loop in `AcervoManager.withComponentAccess` no longer silently no-ops.
- **`Acervo.ensureComponentReady` wired before weight loading** (REQ-PIPE-01, S3 `de8212c`) — `DiffusionPipeline.loadModels(progress:)` now calls `ensureComponentReady` per segment (via `ComponentReadinessService` seam) before invoking `WeightLoader.load`. First-run cache misses auto-download instead of throwing `componentNotDownloaded`.
- **`MemoryManager.hardValidate()` gate in load path** (REQ-PIPE-02, S4 `0c58bf5`) — single up-front `hardValidate(peakMemoryBytes)` call at entry to `loadModels(progress:)` via the `memoryGate` seam. Throws `PipelineError.insufficientMemory(required:available:component:)` on budget exhaustion. Phased `softCheck` per phase is deferred. `MemoryGuardTests.swift` covers both failure and pass-through paths.
- **Role-based component-id lookup replaces positional indexing** (REQ-PIPE-03, S5 `405168e`) — `_allComponentIds: [String]` replaced with `_componentIdByRole: [PipelineRole: String]`; `findComponentId(for:)` is now a dictionary lookup keyed by `PipelineRole`. `PipelineRecipe` protocol gains `componentIdFor: [PipelineRole: String]` with a default implementation that preserves the previous positional convention. Eliminates silent mis-association for recipes that emit IDs in a non-canonical order.
- **End-to-end `withComponentAccess` integration tests** (REQ-INT-01, S6 `bf761d0`) — new `WeightLoaderIntegrationTests.swift` and `ComponentIntegrityTests.swift` in `TuberiaCatalogTests` cover: happy path, integrity failure (`AcervoError.integrityCheckFailed`), not-downloaded failure, and LoRA `withLocalAccess`. All tests use synthetic tensors; no network or real CDN weights.
- **CDN manifest cross-check in CI** (REQ-CDN-01, S7 `f4e6939`) — new `VerifyComponentManifest` SwiftPM executable target (`Tools/VerifyComponentManifest/main.swift`) compares per-file `{path, size, sha256}` from a downloaded `manifest.json` against `CatalogRegistration` descriptors; exits non-zero on any divergence. Step added to `.github/workflows/ensure-model-cdn.yml` after `Verify upload`.

### v0.3.7
- **SwiftAcervo floor bumped to 0.7.2** (REQ-T5) — enables v2 API access (`withComponentAccess`, `ComponentFile.sha256`, `ComponentHandle`, `Acervo.register`, `Acervo.ensureComponentReady`, `AcervoError.integrityCheckFailed`). Package.resolved lock remains consistent.

### v0.3.6
- **DiffusionPipeline**: Cast backbone output to float32 before scheduler math — prevents float16 rounding error amplification (~157×) at high-noise timesteps, fixing channel-specific bias accumulation in PixArt fixtures.
- **WeightLoader**: Move MACF bypass before empty-safetensors guard — ensures the bypass fires before any early-exit check.
- **BetaSchedule**: Add `scaledLinear` schedule variant; T5 int4 dequantization support.

### v0.3.5
- **WeightLoader**: Removed `canEnumerateDirectory` guard from MACF bypass — `fopen()` is blocked by MACF even when directory enumeration succeeds, so the bypass now fires unconditionally for App Group Container paths.

### v0.3.4
- **WeightLoader/T5XXLEncoder**: Require explicit `VINETAS_TEST_MODELS_DIR` for integration tests — tests skip cleanly rather than failing when the env var is absent.

### v0.3.3
- **T5XXLEncoder**: MACF-aware tokenizer directory redirect — resolves tokenizer path through App Group Container when direct access is blocked by Managed App Configuration Framework.

### v0.3.2
- **WeightLoader bug fix**: Hardlink bypass for App Group Container now only activates when the process cannot actually enumerate the directory (`canEnumerateDirectory` check). Prevents stale `/tmp` hardlinks from shadowing real model files in the entitled Vinetas app. Root cause: SwiftAcervo 0.5.6's `withModelAccess()` fallback correctly resolved short component IDs (e.g. `"t5-xxl-encoder-int4"`) to their App Group Container path, exposing the over-broad bypass condition.

### v0.3.1
- **VAE weight fix**: Removed double-transpose in `tensorTransform` — SDXL VAE weights are already in MLX format.

### v0.3.0
- **Test coverage remediation**: removed 6 redundant tests; added 3 new test files (`DPMSolverSchedulerTests`, `FlowMatchEulerSchedulerTests`, `ImageRendererUnitTests`) and 3 new suites (`SDXLVAEDecoderTensorTransformTests`, `LoRAKeyConventionTests`, `PipelineErrorTests`); strengthened `applyWeightsDoesNotCrash`; added edge-case tests to `DeviceCapabilityTests`.
- **WeightLoader refactor**: `loadFromPath()` now `async throws`; routes all local file I/O through `AcervoManager.withLocalAccess` (SwiftAcervo 0.6.0). Tuberia no longer calls `FileManager` directly.
- **DiffusionPipeline**: `generate()` now guards `encoder.isLoaded`, `backbone.isLoaded`, `decoder.isLoaded` at entry, throwing `PipelineError.missingComponent(role:)` if any segment is unloaded.
- **MockPipelineRecipe**: added `loaded()` test factory; test weight loading moved to `DiffusionPipeline+TestSupport.swift` extension.
- **LoRA interop docs**: AGENTS.md and README.md now document the `base_model.model.` prefix gap for PEFT/flux-2 adapters and downstream package interop status.
- Bumped `SwiftAcervo` to `0.6.0`.

### v0.2.7
- Patch release — documentation and organizational update (AGENTS.md/CLAUDE.md/GEMINI.md).

### v0.2.6
- **Bug fix**: Replaced `MLXNN.silu()` compiled op with direct `h * MLX.sigmoid(h)` in `SDXLVAEModel` (`ResnetBlock2D` ×2, `SDXLVAEDecoderModel` ×1). Avoids compiled-op issues while producing identical results.
- Files changed: `Sources/TuberiaCatalog/Decoders/SDXLVAEModel.swift`

### v0.2.5
- Bumped `SwiftAcervo` minimum to 0.5.5

---

## App Group configuration (required)

This package depends on [SwiftAcervo](https://github.com/intrusive-memory/SwiftAcervo) for shared model storage. SwiftAcervo v0.10.0 resolves its App Group ID in this order: `ACERVO_APP_GROUP_ID` env var → `com.apple.security.application-groups` entitlement (macOS only) → `fatalError`. There is **no silent fallback**.

- **Signed UI apps (macOS / iOS)**: declare `com.apple.security.application-groups` with `group.intrusive-memory.models` in your `.entitlements` file. iOS apps additionally need `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the launch environment.
- **CLI tools, scripts, CI jobs, test runners**: export `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the shell or job environment. The standard place is `~/.zprofile`:

    ```sh
    export ACERVO_APP_GROUP_ID=group.intrusive-memory.models
    ```

Without this, `Acervo.sharedModelsDirectory` traps with `fatalError`. See [SwiftAcervo's USAGE.md](https://github.com/intrusive-memory/SwiftAcervo/blob/main/USAGE.md) for full details.

---

## Project Overview

SwiftTuberia ("tuberia" -- Spanish for "plumbing" or "piping system") is a componentized generation pipeline for MLX inference. It provides typed pipe segment protocols, a diffusion pipeline compositor, shared component catalog, and infrastructure services.

**Key concept**: Each pipeline component is a typed pipe segment with a defined inlet and outlet. Model-specific packages provide only their unique backbone architecture and a recipe declaring which pipes to connect. Everything else comes from the shared catalog.

## Architecture

Two products:
- **Tuberia** -- Protocols + infrastructure (TextEncoder, Scheduler, Backbone, Decoder, Renderer protocols; WeightLoader, MemoryManager, LoRA, Progress)
- **TuberiaCatalog** -- Concrete shared components (T5-XXL, SDXL VAE, DPM-Solver++, FlowMatch Euler, ImageRenderer, AudioRenderer)

Documentation:
- [REQUIREMENTS.md](REQUIREMENTS.md) — active mission scope
- [GENERATION_PATHS.md](docs/GENERATION_PATHS.md) — generation path analysis

## Dependencies

- `mlx-swift` (≥ 0.31.3) -- Apple's MLX framework for Swift
- `swift-tokenizers` (≥ 0.4.2, `Swift` trait) -- Tokenizer support (`Tokenizers` product only)
- `SwiftAcervo` (≥ 0.8.3) -- Model registry, CDN download, integrity verification

## Platform Requirements

- iOS 26.0+, macOS 26.0+ exclusively
- Swift 6.2+, Xcode 26+
- Apple Silicon only (M1+)

## Build and Test

```bash
xcodebuild build -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'
xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO
```

Available schemes: `SwiftTuberia-Package`, `Tuberia`, `TuberiaCatalog`. Run `xcodebuild -list` to confirm. **Never use `-scheme SwiftTuberia`** — that scheme does not exist.

### Always disable parallel test execution

`xcodebuild test` must be invoked with `-parallel-testing-enabled NO`. MLX owns a process-global Metal GPU stream, and Swift Testing's default cross-suite parallelism races on the shared command buffer, producing `-[_MTLCommandBuffer addCompletedHandler:] 'Completed handler provided after commit call'` and aborting the process. Suite-level `.serialized` traits only serialize tests within a suite — they do not prevent sibling suites from running concurrently. `make test` and the CI workflow already apply this flag; any ad-hoc `xcodebuild test` invocation must set it too.

See [REQUIREMENTS.md](REQUIREMENTS.md) for the complete specification.
See [GENERATION_PATHS.md](docs/GENERATION_PATHS.md) for generation path analysis.

---

## LoRA Adapter Interoperability

SwiftTuberia's `LoRALoader` (`Sources/Tuberia/Pipeline/LoRALoader.swift`) is the shared LoRA merge engine used by all downstream model packages. Maintaining compatibility with adapters trained for `pixart-swift-mlx` and `flux-2-swift-mlx` is a **first-class priority**.

### Supported Key Conventions

| Convention | Example key | Status |
|-----------|-------------|--------|
| HuggingFace suffix | `layer.weight.lora_A` / `.lora_B` | ✅ Supported |
| HuggingFace dot-weight | `layer.lora_A.weight` / `.lora_B.weight` | ✅ Supported |
| Diffusers down/up | `layer.lora_down` / `.lora_up` (and `.weight` variants) | ✅ Supported |
| PEFT `base_model.model.` prefix | `base_model.model.layer.lora_A.weight` | ❌ **Not supported** — see below |
| `transformer.` prefix (Diffusers) | `transformer.layer.lora_A.weight` | ⚠️ Passes through unstripped; works only if backbone `keyMapping` already expects the prefix |
| `unet.` prefix | `unet.layer.lora_A.weight` | ❌ **Not supported** |

### Known Gap: `base_model.model.` Prefix (PEFT/flux-2 adapters)

Adapters trained with PEFT (the dominant training framework for Flux-based models) emit keys prefixed with `base_model.model.`. Example:

```
base_model.model.double_blocks.0.img_attn.proj.lora_A.weight
```

`LoRALoader.parseLoRAKey` strips the LoRA suffix (`.lora_A.weight`) but does **not** strip `base_model.model.`. This means PEFT-trained adapters targeting Flux-2 will silently fail to merge — every key in the adapter will find no match in the base model parameters.

`flux-2-swift-mlx` handles this locally in its own `LoRALoader.swift` (line 132: `layerPath.hasPrefix("base_model.model.")`). Until SwiftTuberia adds the same prefix stripping, `flux-2-swift-mlx` adapters loaded via the shared `LoRALoader` will not apply correctly.

**Before adding any LoRA loading path that bypasses the local Flux-2 loader, add `base_model.model.` prefix stripping to `LoRALoader.parseLoRAKey`.**

### Known Gap: `unet.` Prefix

Older SD-style adapters sometimes prefix keys with `unet.`. This convention is documented but not implemented. If you add support, add a test in `LoRAKeyConventionTests` in `Tests/TuberiaGPUTests/LoRATests.swift`.

### Interop Test Policy

Any new key convention added to `LoRALoader` **must** have a corresponding test in `Tests/TuberiaGPUTests/LoRATests.swift` in the `LoRAKeyConventionTests` suite. The suite uses synthetic `ModuleParameters` — no real adapter files required.

## Telemetry

SwiftTuberia ships a **dual-seam telemetry surface** that lets hosts observe every
boundary event in the diffusion pipeline without coupling to internal types. The
surface has four parts: `TuberiaTelemetryEvent` (typed event enum, 27 cases),
`TuberiaTelemetryReporter` (Sendable async-capture protocol), an instance-bound
seam on `DiffusionPipeline` for test isolation, and a process-wide seam via
`TuberiaTelemetry` for CLI hosts that can't reach pipeline instances directly. A
contributor would touch this surface to add a new observable operation, to add an
adapter in a downstream CLI, or to extend the test matrix for the dual-seam
priority rules.

### Subscribing as a host (process-wide)

```swift
// At CLI / process startup:
TuberiaTelemetry.setReporter(myAdapter)

// At shutdown:
TuberiaTelemetry.setReporter(nil)
```

`TuberiaTelemetry.setReporter(_:)` is thread-safe (backed by
`OSAllocatedUnfairLock`). All current and future `DiffusionPipeline` instances
emit to this reporter whenever no instance-bound reporter is installed.

### Per-instance subscription (test isolation)

```swift
let pipeline = DiffusionPipeline(recipe: recipe)
pipeline.setTelemetry(myMockReporter)  // instance wins over process-wide
// … run assertions …
pipeline.setTelemetry(nil)             // detach; process-wide reporter (if any) resumes
```

Instance reporter always takes priority. Tests that install a mock via
`setTelemetry(_:)` are not polluted by an ambient process-wide reporter installed
by the test harness or a sibling test.

### Adding a new event case

1. **Add the case** to `TuberiaTelemetryEvent` in
   `Sources/Tuberia/Telemetry/TuberiaTelemetryEvent.swift`. All associated-value
   types must be `Sendable`. Do not add a `runID` field — that belongs at the
   host/sink envelope layer, not on the event itself.
2. **Emit it** at exactly one canonical call site via
   `await self.telemetry?.capture(.yourNewEvent(...))` (using `telemetry`, the
   computed property that applies the instance-wins-over-process-wide rule).
   Never call `_instanceReporter?.capture(...)` or `TuberiaTelemetry.current?.capture(...)`
   directly.
3. **Update the host adapter** (`TuberiaTelemetryAdapter` in SwiftVinetas) — it
   switches exhaustively over `TuberiaTelemetryEvent`. Adding a case without
   updating the adapter causes a compile error in the host; this is intentional.
4. **Add tests** in `Tests/TuberiaTests/` asserting the new case fires for the
   instance-bound seam and the process-wide seam. Tear down with
   `TuberiaTelemetry.setReporter(nil)` in `tearDown` to prevent cross-test leakage.

### When to add a new event

- **Single canonical emission site.** Each case fires in exactly one place in source.
  Multiple emission sites for the same logical event are a sign the event should be
  split or the call sites should be refactored.
- **All payload types must be `Sendable`.** Wrapping a non-Sendable value in a
  struct to make it compile is not acceptable — redesign the payload.
- **Do not add `runID`.** Hosts attach run identifiers in the envelope when
  serialising; the event enum must remain host-agnostic.
- **Prefer `start`/`complete` pairs.** For any operation with measurable duration,
  emit both a `...Start` and `...Complete` case. Single events are appropriate only
  for point-in-time facts (e.g. `memoryGateChecked`, `componentReadinessChecked`).
- **Honor hot-path discipline.** Emission sites inside the denoise loop must not
  trigger synchronous MLX tensor reads (`.toArray()`, scalar extractions) except
  where documented. `TuberiaTensorStat.sample(...)` is the approved sampling helper;
  its tensor reads are already accounted for in the ≤1% overhead budget.

For the cross-library dual-seam pattern, see `SwiftVinetas/docs/INSTRUMENTATION_PATTERN.md`.

---

## Common Agent Tasks

- **Bug fix in a decoder/encoder**: Read the relevant file under `Sources/TuberiaCatalog/`, make the change, build with `xcodebuild`, verify via tests.
- **Adding a new catalog component**: Implement the protocol from `Tuberia` target, place under `Sources/TuberiaCatalog/`, add to `TuberiaCatalog.swift` exports.
- **Updating dependencies**: Edit `Package.swift`, run `xcodebuild -resolvePackageDependencies`.
- **Releasing**: Follow the ship-swift-library skill workflow — bump version on `development`, merge PR, tag on `main`.

## Critical Rules for AI Agents

1. NEVER commit directly to `main`
2. ONLY supports iOS 26.0+ and macOS 26.0+ — NEVER add code for older platforms
3. NEVER use `swift build` or `swift test` — always `xcodebuild` (or XcodeBuildMCP)
4. ALWAYS read files before editing
5. NEVER create files unless necessary
6. Follow agent-specific instructions — see [CLAUDE.md](CLAUDE.md) or [GEMINI.md](GEMINI.md)
