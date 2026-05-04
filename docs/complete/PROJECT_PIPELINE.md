# PROJECT PIPELINE — Unified MLX Inference Architecture

**Status**: DRAFT — debate and refine before implementation.
**Date**: 2026-03-21
**Scope**: Cross-repository architectural plan to unify all MLX inference behind a single componentized pipeline system. Five repositories, one coherent stack.

---

## The Problem

Every MLX model we ship rebuilds the same infrastructure from scratch — weight loading, quantization, memory management, model downloading, scheduling, image rendering. The model-specific code (the unique neural network) is ~20% of each library. The other ~80% is plumbing.

Adding a new model means building a full standalone library. Adding a new output modality (video, non-speech audio) means even more duplication. The per-model cost is too high.

## The Solution

**SwiftTubería** (*tubería* — Spanish for "plumbing" or "piping system"): a componentized generation stack where each piece is a typed pipe segment with a defined inlet and outlet. The name is literal, not metaphorical. A building's plumbing has standardized pipe segments with typed connections — a half-inch copper inlet connects to a half-inch copper outlet, always. You test a single segment by running water through it. You swap a corroded segment without re-plumbing the whole building. A plumber assembles from a catalog of standard parts, adding custom fittings only where the building demands something unique.

SwiftTubería works identically: model-specific packages provide only the custom fitting — their unique backbone architecture and a recipe declaring which pipes to connect. Everything else comes from the shared catalog and infrastructure.

**SwiftAcervo v2**: the model registry. Model plugins register what exists; Acervo manages the full lifecycle — catalog, download, cache, integrity verification, abstracted access. Pipeline code never touches file paths.

---

## Architecture Overview

```
                          Consumer Applications
                    ┌───────────┐  ┌────────────┐
                    │Produciesta│  │  iPad App   │
                    └─────┬─────┘  └──────┬─────┘
                          │               │
                   ┌──────▼───────────────▼──────┐
                   │        SwiftVinetas          │  Domain: prompt composition,
                   │   (EngineRouter, Engines)    │  style management, characters
                   └──────┬───────────────┬──────┘
                          │               │
              ┌───────────▼───┐   ┌───────▼──────────┐
              │  Flux2Engine  │   │  PixArtEngine     │  PixArtEngine: ~50 lines,
              │  (unchanged,  │   │  (thin adapter)   │  assembles recipe via
              │   wraps       │   │                   │  SwiftTubería pipeline
              │  Flux2Pipeline│   │                   │
              │   directly)   │   │                   │
              └───────┬───────┘   └────────┬──────────┘
                      │                    │
                      ▼                    │
              ┌──────────────┐             │
              │flux-2-swift- │             │
              │mlx (unchanged│             │
              │  standalone) │             │
              └──────────────┘             │
         ┌─────────────────────────────────▼─────────────┐
         │             SwiftTubería                   │
         │                                                │
         │  ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
         │  │ Pipeline  │ │ Catalog  │ │ Infrastructure │ │
         │  │ Protocols │ │ T5-XXL   │ │ WeightLoader   │ │
         │  │ Diffusion │ │ SDXL VAE │ │ MemoryManager  │ │
         │  │ Pipeline  │ │ DPM-Solv │ │ DeviceCapab.   │ │
         │  │ Recipes   │ │ FlowMatch│ │ LoRA Loader    │ │
         │  │           │ │ Renderer │ │ Progress       │ │
         │  └──────────┘ └──────────┘ └────────────────┘ │
         └───────────────────────┬────────────────────────┘
                                 │
         ┌───────────────────────▼────────────────────────┐
         │              SwiftAcervo v2                     │
         │  Component Registry + Download + Integrity      │
         │  ~/Library/SharedModels/ (zero dependencies)    │
         └────────────────────────────────────────────────┘
```

### Model Plugin Packages (provide ONLY the delta)

```
┌─────────────────────┐    ┌─────────────────────┐
│   pixart-swift-mlx  │    │    SwiftVoxAlta     │
│                     │    │                     │
│ - PixArt DiT        │    │ - VoiceProvider     │
│   backbone          │    │ - TTS generation    │
│ - Weight key map    │    │   (mlx-audio-swift) │
│ - Config            │    │ - Voice identity    │
│ - Recipe            │    │ - Acervo registry   │
│ - Acervo descriptors│    │   entries           │
│                     │    │ - Infrastructure    │
│ ~400 lines new code │    │   adoption only     │
└─────────────────────┘    └─────────────────────┘
```

> **Note**: flux-2-swift-mlx remains a standalone library. FLUX.2 migration to SwiftTubería is deferred — it already works, and PixArt validates the architecture first. Flux2Engine in SwiftVinetas continues wrapping Flux2Pipeline directly.

---

## Dependency Graph

```
pixart-swift-mlx ──▶ SwiftTubería ──▶ SwiftAcervo
SwiftVinetas ──────▶ SwiftTubería ──▶ SwiftAcervo
SwiftVoxAlta ──────▶ SwiftTubería ──▶ SwiftAcervo
                                          (zero deps)

flux-2-swift-mlx ── (unchanged, standalone — no pipeline dependency)
```

SwiftAcervo is the leaf — zero external dependencies, Foundation only.
SwiftTubería depends on mlx-swift, swift-transformers, and SwiftAcervo.
Everything else depends inward. flux-2-swift-mlx remains standalone.

---

## Per-Repository Summary

### 1. SwiftAcervo — Component Registry

**Full spec**: `SwiftAcervo/REQUIREMENTS.md`

Evolves from filesystem-only discovery ("what's on disk?") to declarative component registry ("what exists in the world?"). Model plugins register `ComponentDescriptor` entries at import time. Acervo manages the full lifecycle.

**Key additions**:
- `ComponentDescriptor` — declares a downloadable component (HF repo, files, sizes, SHA-256 checksums)
- `ComponentHandle` — scoped, abstracted file access (no path leakage)
- `Acervo.ensureComponentReady(id)` — registry-aware downloads (caller no longer specifies file lists)
- SHA-256 integrity verification on download and before access
- Backward compatible — all v1 API unchanged, registry is additive

**Design principle**: Pipeline code never constructs file paths. All model access through Acervo abstractions.

---

### 2. SwiftTubería — The Pipe System

**Full spec**: `SwiftTubería/REQUIREMENTS.md`

The centerpiece. Provides typed pipe segment protocols, a diffusion pipeline compositor, shared component catalog, and infrastructure services.

**Component Protocols** (the pipe segments):

| Protocol | Inlet | Outlet |
|---|---|---|
| `TextEncoder` | text + maxLength | embeddings [B, seq, dim] + mask |
| `Scheduler` | config (steps, guidance, betas) | timestep plan; per-step: model output → denoised sample |
| `Backbone` | latents + conditioning + timestep | noise prediction |
| `Decoder` | latents (MLXArray) | decoded data + metadata |
| `Renderer` | decoded output | final format (CGImage, WAV Data, etc.) |

Each connection has a **shape contract** validated at pipeline assembly time, not generation time.

**Shared Component Catalog**:
- Encoders: T5-XXL, CLIP (future)
- Schedulers: DPM-Solver++, Flow-Match Euler, DDPM
- Decoders: SDXL VAE
- Renderers: ImageRenderer, AudioRenderer

**Infrastructure**:
- WeightLoader (loads through Acervo ComponentHandle, never file paths)
- MemoryManager (device detection, budget enforcement, two-phase coordination)
- LoRA infrastructure (load/apply/scale/unload — matches LoRA keys to loaded model keys)
- Progress reporting (unified across all pipeline operations)

**Two products**: `Tubería` (protocols + infrastructure) and `TuberíaCatalog` (concrete shared components).

---

### 3. pixart-swift-mlx — PixArt Model Plugin

**Full spec**: `pixart-swift-mlx/REQUIREMENTS.md`

Provides the PixArt-Sigma DiT backbone — 28 blocks, 1152 hidden dim, ~600M parameters. The only substantial new code.

**Pipeline recipe**:
```
T5XXLEncoder (catalog) → PixArtDiT (this repo) → SDXLVAEDecoder (catalog) → ImageRenderer (catalog)
                              ▲
                       DPMSolver++ (catalog)
```

**Key facts**: ~2 GB total (int4), iPad-viable, Apache 2.0 licensed, ~400 lines of model-specific code.

Also provides: weight key mapping (~200 keys), Acervo component descriptors, LoRA target declarations, weight conversion scripts, CLI tool.

---

### 4. SwiftVoxAlta — Infrastructure Adoption

**Full spec**: `SwiftVoxAlta/REQUIREMENTS.md`

**Infrastructure adoption, not pipeline migration.** TTS is autoregressive, not diffusion — the generation path through mlx-audio-swift is unchanged. What changes:

- Model management → Acervo v2 Component Registry (6 TTS model variants registered)
- Memory management → SwiftTubería's MemoryManager
- Device detection → SwiftTubería's DeviceCapability

**What stays unchanged**: VoiceProvider protocol, voice clone prompts, VoiceLock/VoiceCache, .vox file handling, GenerationSettings, diga CLI, all domain-specific logic.

---

### 5. SwiftVinetas — Engine Simplification

**Full spec**: `SwiftVinetas/REQUIREMENTS.md`

PixArtEngine comes alive (replacing the stub) as ~50 lines wrapping a composed `DiffusionPipeline`. Flux2Engine stays unchanged, wrapping `Flux2Pipeline` directly. iPad deployment enabled via PixArt.

**What stays unchanged**: VinetasClient public API, EngineRouter, ModelDescriptor protocol, StyleConfig, PromptFile, AspectRatio, character management, prompt composition, Flux2Engine implementation. Consumers see zero behavioral change.

**Platform strategy**: Both engines on macOS, PixArt-only on iPad (memory-gated).

---

## Execution Order

The work is organized into **waves**. Each wave produces shippable, testable artifacts. Later waves depend on earlier ones but repositories within a wave can often be worked in parallel.

### Wave 0 — Foundation (no model-specific code)

| Step | Repository | Work | Depends On | Validates |
|---|---|---|---|---|
| 0.1 | **SwiftAcervo** | Component Registry types (`ComponentDescriptor`, `ComponentType`, `ComponentFile`) | — | Types compile, unit tests pass |
| 0.2 | **SwiftAcervo** | Registration API (`register`, `unregister`, catalog queries) | 0.1 | Register/query/deduplicate |
| 0.3 | **SwiftAcervo** | Registry-aware downloads (`ensureComponentReady`) | 0.2 | Download uses registry file lists |
| 0.4 | **SwiftAcervo** | ComponentHandle + `withComponentAccess` | 0.3 | Scoped access, no path leakage |
| 0.5 | **SwiftAcervo** | SHA-256 integrity verification | 0.4 | Corrupt files rejected |

**Milestone**: SwiftAcervo v2 shipped. All downstream work depends on this.

### Wave 1 — Pipeline Core (protocols + infrastructure)

| Step | Repository | Work | Depends On | Validates |
|---|---|---|---|---|
| 1.1 | **SwiftTubería** | Pipe segment protocols (TextEncoder, Scheduler, Backbone, Decoder, Renderer) | Wave 0 | Protocols compile, mock conformances |
| 1.2 | **SwiftTubería** | Infrastructure: WeightLoader (loads through ComponentHandle) | 0.4 | Load safetensors via Acervo handle |
| 1.3 | **SwiftTubería** | Infrastructure: MemoryManager + DeviceCapability | — | Device detection, budget enforcement |
| 1.4 | **SwiftTubería** | Infrastructure: LoRA loader, Progress reporter | 1.1 | LoRA mechanics, progress enum |
| 1.5 | **SwiftTubería** | DiffusionPipeline compositor + PipelineRecipe protocol | 1.1 | Mock pipeline assembles and validates |
| 1.6 | **SwiftTubería** | Pipeline contract tests (shape validation, incompatible assembly fails) | 1.5 | Mismatched components produce clear errors |

**Milestone**: SwiftTubería compiles with mock components. No real models yet.

### Wave 2 — Shared Components (the catalog)

| Step | Repository | Work | Depends On | Validates |
|---|---|---|---|---|
| 2.1 | **SwiftTubería** | ImageRenderer (MLXArray → CGImage) | 1.1 | Known pixel data → valid CGImage |
| 2.2 | **SwiftTubería** | SDXLVAEDecoder | 1.1, 1.2 | Known latent → pixels (PSNR > 30 dB vs reference) |
| 2.3 | **SwiftTubería** | DPMSolverScheduler | 1.1 | Synthetic noise → expected denoising trajectory |
| 2.4 | **SwiftTubería** | T5XXLEncoder | 1.1, 1.2 | Known prompt → expected embedding shape |
| 2.5 | **SwiftTubería** | FlowMatchEulerScheduler | 1.1 | Flow matching step math validates |
| 2.6 | **SwiftTubería** | AudioRenderer (MLXArray → WAV Data) | 1.1 | Known samples → valid WAV |

Steps 2.1–2.6 are independent of each other and can be parallelized.

**Milestone**: All catalog components tested in isolation. Ready for real model integration.

### Wave 3 — First Model Plugin (proves the system end-to-end)

| Step | Repository | Work | Depends On | Validates |
|---|---|---|---|---|
| 3.1 | **pixart-swift-mlx** | PixArt DiT backbone (Backbone protocol conformance) | 1.1 | Forward pass with synthetic input → expected shape |
| 3.2 | **pixart-swift-mlx** | Weight key mapping + config | 3.1 | Weights load and map correctly |
| 3.3 | **pixart-swift-mlx** | Acervo component descriptors registered | Wave 0 | Components discoverable and downloadable |
| 3.4 | **pixart-swift-mlx** | Pipeline recipe (T5 + DPM + PixArtDiT + SDXL VAE + ImageRenderer) | 3.1, Wave 2 | Recipe assembles, validation passes |
| 3.5 | **pixart-swift-mlx** | End-to-end: prompt → CGImage | 3.4 | Real image generated, correct dimensions |
| 3.6 | **pixart-swift-mlx** | Weight conversion scripts | 3.1 | PyTorch → MLX safetensors, PSNR > 30 dB |
| 3.7 | **pixart-swift-mlx** | PixArtCLI | 3.5 | CLI generates images from command line |

**Milestone**: First model runs through the pipeline end-to-end. Architecture is validated.

### Wave 4 — Consumer Integration

| Step | Repository | Work | Depends On | Validates |
|---|---|---|---|---|
| 4.1 | **SwiftVinetas** | PixArtEngine (real implementation, ~50 lines) | Wave 3 | PixArt generates through engine layer |
| 4.2 | **SwiftVinetas** | iPad deployment (platform-conditional engine registration) | 4.1 | PixArt-only on iPadOS |
| 4.3 | **SwiftVoxAlta** | Acervo v2 adoption (register TTS component descriptors) | Wave 0 | TTS models in registry |
| 4.4 | **SwiftVoxAlta** | MemoryManager + DeviceCapability adoption | 1.3 | Unified memory tracking |
| 4.5 | **SwiftVoxAlta** | Remove hardcoded file lists and path construction | 4.3 | All model knowledge in descriptors |

Flux2Engine remains unchanged — it continues wrapping `Flux2Pipeline` directly.

Steps 4.1–4.2 (Vinetas) and 4.3–4.5 (VoxAlta) are independent and can be parallelized.

**Milestone**: All consumers integrated. Architecture fully deployed.

---

## Critical Path

The longest dependency chain determines the minimum timeline:

```
SwiftAcervo v2 (Wave 0)
    → SwiftTubería protocols + infrastructure (Wave 1)
        → Catalog components (Wave 2, parallelizable)
            → PixArt plugin (Wave 3, first end-to-end)
                → SwiftVinetas PixArtEngine (Wave 4.1)
                    → iPad deployment (Wave 4.2)
```

VoxAlta integration (Wave 4.3–4.5) can start as soon as Wave 0 is done and runs in parallel with all other waves.

```
                    ┌─ Wave 2 (catalog, parallel) ─── Wave 3 (PixArt) ─── Wave 4.1-4.2 (Vinetas)
Wave 0 (Acervo) ──▶│
                    ├─ Wave 1 (Pipeline core)
                    │
                    └─ Wave 4.3-4.5 (VoxAlta, independent)
```

---

## What Success Looks Like

1. **Adding a new diffusion model** = backbone (~300 lines) + config (~30 lines) + key mapping (~50 lines) + recipe (~20 lines). That's it.

2. **Shared components tested once**, validated for every model. Fix a bug in SDXLVAEDecoder → fixed for PixArt and every future model using it.

3. **Pipe segments independently testable** with synthetic inputs. No full pipeline needed to validate a component.

4. **Pipeline assembly validates compatibility** at construction time. Mismatched components produce clear errors, not silent corruption.

5. **All model access through Acervo**. No file paths in pipeline code. Storage is an implementation detail.

6. **iPad image generation** — PixArt runs on 8 GB M-series iPads. SwiftVinetas becomes a macOS + iPad library.

7. **Clear path for future migrations** — FLUX.2 and future models can migrate to the pipeline when ready. The architecture is proven by PixArt; migration is retrofitting, not experimentation.

8. **Clear path for future modalities** — Video and non-speech audio diffusion are just new backbones + recipes. The pipeline, catalog, and infrastructure are already there.

---

## Repository Index

| Repository | Role | REQUIREMENTS.md |
|---|---|---|
| `SwiftAcervo` | Component Registry + model management | [`SwiftAcervo/REQUIREMENTS.md`](SwiftAcervo/REQUIREMENTS.md) |
| `SwiftTubería` | Pipe protocols + catalog + infrastructure | [`SwiftTubería/REQUIREMENTS.md`](SwiftTubería/REQUIREMENTS.md) |
| `pixart-swift-mlx` | PixArt-Sigma model plugin | [`pixart-swift-mlx/REQUIREMENTS.md`](pixart-swift-mlx/REQUIREMENTS.md) |
| `SwiftVoxAlta` | TTS — infrastructure adoption | [`SwiftVoxAlta/REQUIREMENTS.md`](SwiftVoxAlta/REQUIREMENTS.md) |
| `SwiftVinetas` | Engine layer — pipeline consumer | [`SwiftVinetas/REQUIREMENTS.md`](SwiftVinetas/REQUIREMENTS.md) |

> **Out of scope**: `flux-2-swift-mlx` remains a standalone library. FLUX.2 migration to SwiftTubería is deferred until the pipeline architecture is validated by PixArt.

Each repository's REQUIREMENTS.md contains the full specification for that repo's work. This document provides the unified view and execution order.
