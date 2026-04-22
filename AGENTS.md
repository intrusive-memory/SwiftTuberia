# AGENTS.md

This file provides comprehensive documentation for AI agents working with the SwiftTuberia codebase.

**Version**: 0.5.0

---

## Recent Changes

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

## Project Overview

SwiftTuberia ("tuberia" -- Spanish for "plumbing" or "piping system") is a componentized generation pipeline for MLX inference. It provides typed pipe segment protocols, a diffusion pipeline compositor, shared component catalog, and infrastructure services.

**Key concept**: Each pipeline component is a typed pipe segment with a defined inlet and outlet. Model-specific packages provide only their unique backbone architecture and a recipe declaring which pipes to connect. Everything else comes from the shared catalog.

## Architecture

Two products:
- **Tuberia** -- Protocols + infrastructure (TextEncoder, Scheduler, Backbone, Decoder, Renderer protocols; WeightLoader, MemoryManager, LoRA, Progress)
- **TuberiaCatalog** -- Concrete shared components (T5-XXL, SDXL VAE, DPM-Solver++, FlowMatch Euler, ImageRenderer, AudioRenderer)

Documentation:
- [REQUIREMENTS.md](REQUIREMENTS.md) — active mission scope
- [GENERATION_PATHS.md](GENERATION_PATHS.md) — generation path analysis

## Dependencies

- `mlx-swift` -- Apple's MLX framework for Swift
- `swift-transformers` -- Tokenizer support
- `SwiftAcervo` -- Model registry and download management

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
See [GENERATION_PATHS.md](GENERATION_PATHS.md) for generation path analysis.

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
