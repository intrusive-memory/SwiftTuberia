# SwiftTuberia — Requirements (Overview)

**Status**: DRAFT — debate and refine before implementation.
**Parent project**: [`PROJECT_PIPELINE.md`](../PROJECT_PIPELINE.md) — Unified MLX Inference Architecture (§2. SwiftTuberia, Waves 1–2)
**Scope**: Unified, componentized generation stack for MLX inference on Apple Silicon. All model-specific packages (pixart-swift-mlx, flux-2-swift-mlx, future video/audio models) plug into this pipeline rather than building standalone stacks.

---

## Motivation

Every MLX inference workflow follows the same pattern:

```
Condition(prompt) → Generate(latents, condition, timesteps) → Decode(raw output) → Render(final format)
```

Today, each model repo (flux-2-swift-mlx, pixart-swift-mlx, SwiftVoxAlta) rebuilds this entire stack from scratch — model downloading, weight loading, quantization, memory management, scheduling, VAE decoding, image rendering. The model-specific code (the unique neural network architecture) is typically ~20% of each library; the other ~80% is shared infrastructure rebuilt each time.

SwiftTuberia inverts this. It provides the pipeline system and shared components. Model packages provide only the delta — their unique backbone architecture, weight key mapping, and a recipe that declares which pipe segments to connect.

### Design Metaphor: Tuberia (Plumbing)

*Tuberia* is Spanish for "plumbing" or "piping system" — the network of pipes that carries water from source to destination. This is not an analogy. It is a literal description of what this library does.

A building's plumbing system has standardized pipe segments with typed connections. A half-inch copper inlet connects to a half-inch copper outlet — always. You can test a single pipe segment by running water through it without connecting the entire system. You can swap a corroded segment for a new one without re-plumbing the whole building. A plumber assembles a system from a catalog of standard parts, adding custom fittings only where the building demands something unique.

SwiftTuberia works the same way:

- **Pipe segments** are typed components (TextEncoder, Scheduler, Backbone, Decoder, Renderer) with defined inlets and outlets
- **Recipes** are blueprints — they declare which pipe segments connect to form a complete system
- **The catalog** is the hardware store — standard components (T5-XXL encoder, SDXL VAE decoder, DPM-Solver scheduler) that any model can pull off the shelf
- **Model plugins** provide only the custom fitting — the unique backbone architecture that is specific to one model. Everything else comes from the catalog.
- **Assembly-time validation** ensures every connection is compatible before anything flows. A 4096-dim encoder outlet connected to a 768-dim backbone inlet fails at assembly, not at generation.

The plumbing metaphor also explains SwiftVoxAlta's relationship: TTS doesn't use the pipe system (it's autoregressive, not diffusion), but it does connect to the **infrastructure** — the water meter (MemoryManager), the pressure regulator (DeviceCapability), and the utility company (SwiftAcervo). You don't need pipes to benefit from shared infrastructure.

```
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│  Text     │───▶│  Backbone  │───▶│  Decoder  │───▶│ Renderer  │
│  Encoder  │    │ (per-model)│    │           │    │           │
│           │    │            │    │           │    │           │
│ inlet:    │    │ inlet:     │    │ inlet:    │    │ inlet:    │
│  String   │    │ Embeddings │    │ MLXArray  │    │ Decoded   │
│ outlet:   │    │ + Latents  │    │ (latents) │    │  Output   │
│ Embeddings│    │ + Timestep │    │ outlet:   │    │ outlet:   │
└──────────┘    │ outlet:    │    │ Decoded   │    │ Final     │
                │  MLXArray  │    │  Output   │    │  Output   │
                └───────────┘    └──────────┘    └──────────┘
```

A diffusion pipeline also includes a **Scheduler** segment that drives the backbone iteratively:

```
┌───────────┐
│ Scheduler  │──── drives iteration ────▶ Backbone
│            │◀─── noise prediction ─────│
│ inlet:     │
│  config    │
│ outlet:    │
│  timesteps │
│  + step()  │
└───────────┘
```

---

## How to Read These Documents

This overview provides the architectural context, package structure, and implementation order. The detailed specifications are split into addendum documents, each self-contained for agents working on that area:

| Document | Content | When to Read |
|---|---|---|
| [`requirements/PROTOCOLS.md`](requirements/PROTOCOLS.md) | Pipe segment protocols (TextEncoder, Scheduler, Backbone, Decoder, Renderer) + WeightedSegment lifecycle + canonical Swift definitions | Implementing any pipe segment, building a model plugin backbone, or understanding inlet/outlet contracts |
| [`requirements/PIPELINE.md`](requirements/PIPELINE.md) | DiffusionPipeline orchestration, PipelineRecipe protocol, two-phase loading, LoRA system, error model, progress reporting + canonical Swift definitions | Implementing the pipeline compositor, building a recipe, or understanding the generation flow |
| [`requirements/CATALOG.md`](requirements/CATALOG.md) | Shared component catalog (T5XXLEncoder, SDXLVAEDecoder, DPMSolverScheduler, etc.) + configuration types + Acervo descriptors + canonical Swift definitions | Implementing a catalog component, or writing a recipe that references catalog components |
| [`requirements/INFRASTRUCTURE.md`](requirements/INFRASTRUCTURE.md) | WeightLoader, MemoryManager, DeviceCapability, Acervo integration + canonical Swift definitions | Implementing infrastructure services, or understanding how weights are loaded and memory is managed |
| [`requirements/TESTING.md`](requirements/TESTING.md) | Testing strategy: component tests, contract tests, integration tests, coverage and CI stability requirements | Writing or reviewing tests for any SwiftTuberia code |

**Canonical Swift definitions**: Each addendum includes a "Canonical Swift Definitions" section with copy-paste-ready protocol/type source. If any prose or pseudocode differs from the canonical Swift, **the canonical Swift governs**.

---

## Platforms

```swift
platforms: [.macOS(.v26), .iOS(.v26)]
```

macOS and iPadOS (M-series) are first-class targets. This enables lightweight models (PixArt, future small models) to run on iPad while heavier models (FLUX.2) remain Mac-only — determined by the model's memory requirements, not by the pipeline.

---

## Package Structure

```swift
products: [
    // Core pipeline system — protocols, composition, infrastructure
    .library(name: "Tuberia", targets: ["Tuberia"]),

    // Shared component catalog — reusable pipe segments
    .library(name: "TuberiaCatalog", targets: ["TuberiaCatalog"]),
]
```

- **`Tuberia`** — Protocols, pipeline builder, infrastructure (weight loader, memory manager, progress). Model access via SwiftAcervo. No model-specific code.
- **`TuberiaCatalog`** — Concrete shared components (T5XXLEncoder, SDXLVAEDecoder, DPMSolverScheduler, FlowMatchEulerScheduler, ImageRenderer, AudioRenderer). Depends on Tuberia.

Model plugins depend on `Tuberia` (for protocols) and optionally on `TuberiaCatalog` (for shared components they want to reuse). Infrastructure-only consumers (e.g., SwiftVoxAlta) import only `Tuberia` — they get MemoryManager and DeviceCapability without pulling in any diffusion components.

---

## Dependencies

```swift
.package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.2"),
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
.package(url: "<SwiftAcervo>", from: "2.0.0"),
```

SwiftTuberia depends on mlx-swift for compute, swift-transformers for tokenization, and SwiftAcervo for all model management. It does NOT depend on any model plugin package — the dependency arrow points inward.

```
pixart-swift-mlx ──▶ SwiftTuberia ──▶ SwiftAcervo
SwiftVinetas ──────▶ SwiftTuberia ──▶ SwiftAcervo
SwiftVoxAlta ──────▶ SwiftTuberia ──▶ SwiftAcervo

Model plugins also depend on SwiftAcervo directly (for component registration).
flux-2-swift-mlx is currently standalone (migration deferred).
```

---

## Model Plugin Contract

What a model plugin package (e.g., pixart-swift-mlx, flux-2-swift-mlx) must provide to participate in the pipeline system.

### Required

1. **Backbone implementation** — A struct/class conforming to the `Backbone` protocol (see [PROTOCOLS.md](requirements/PROTOCOLS.md)). This is the unique neural network architecture. This is the only substantial new code per model.

2. **Model configuration** — A struct declaring the architecture parameters (hidden size, head count, depth, etc.). Weight key mapping and optional tensor transforms are provided through the `WeightedSegment` protocol conformance (`keyMapping`, `tensorTransform`).

3. **Pipeline recipe** — Declares which encoder, scheduler, decoder, and renderer to connect (see [PIPELINE.md](requirements/PIPELINE.md)). References catalog components where available, provides custom components where needed.

4. **Acervo component descriptors** — Declares `ComponentDescriptor` entries for registration into SwiftAcervo's Component Registry (HuggingFace repos, file sizes, checksums).

### Optional

5. **Custom TextEncoder** — Only if the model uses an encoder not in the catalog (e.g., FLUX's Qwen3/Mistral encoders).

6. **Custom Decoder** — Only if the model uses a decoder not in the catalog (e.g., FLUX's specific VAE).

7. **LoRA support** — Model-specific LoRA target layers. The infrastructure for loading/applying LoRA weights is in SwiftTuberia (see [PIPELINE.md](requirements/PIPELINE.md)); the plugin declares which layers accept adapters.

8. **Weight conversion scripts** — Python scripts to convert upstream PyTorch checkpoints to MLX safetensors format.

### What Plugins Do NOT Provide

- Model downloading logic (SwiftAcervo handles this)
- Weight loading mechanics (WeightLoader handles this — see [INFRASTRUCTURE.md](requirements/INFRASTRUCTURE.md))
- Memory management (MemoryManager handles this — see [INFRASTRUCTURE.md](requirements/INFRASTRUCTURE.md))
- Noise scheduling (Scheduler catalog handles this — see [CATALOG.md](requirements/CATALOG.md))
- Image/audio rendering (Renderer catalog handles this — see [CATALOG.md](requirements/CATALOG.md))
- Progress reporting (PipelineProgress handles this — see [PIPELINE.md](requirements/PIPELINE.md))
- Quantization logic (WeightLoader handles this — see [INFRASTRUCTURE.md](requirements/INFRASTRUCTURE.md))

---

## Concurrency Model

- **Pipelines** are actors — one generation at a time per pipeline instance.
- **Pipe segments** (encoders, backbones, decoders) are `Sendable` but assume single-threaded access during forward passes. The pipeline actor serializes access.
- **Renderers** are stateless and freely concurrent.
- **SwiftAcervo's AcervoManager** and **Memory Manager** are actors — safe for concurrent queries from multiple pipelines.
- Multiple pipeline instances can exist simultaneously (e.g., one for image generation, one for audio), each with their own loaded components.

---

## What This Replaces

| Previously In | Now In SwiftTuberia | Stays In Original |
|---|---|---|
| pixart-swift-mlx | Weight loading, quantization, memory management, image rendering, SDXL VAE, DPM-Solver, T5 encoder | PixArt DiT backbone, weight key mapping |
| SwiftVoxAlta | Model management, memory management, device detection | VoiceProvider, clone prompts, .vox handling, TTS pipeline specifics |
| SwiftVinetas | Nothing moves out — Vinetas consumes Pipeline for PixArt; Flux2Engine unchanged | Engine abstraction, prompt composition, style management, PromptFile |

> **Note**: flux-2-swift-mlx migration is deferred. It remains a standalone library.

---

## Implementation Order

1. **Tuberia target** — Protocols, infrastructure (weight loader, memory manager, Acervo integration)
2. **DiffusionPipeline** — Generic diffusion orchestrator
3. **ImageRenderer** — Stateless MLXArray → CGImage (simplest renderer, immediate payoff)
4. **SDXLVAEDecoder** — First shared decoder (used by PixArt, validates decoder protocol)
5. **DPMSolverScheduler** — First shared scheduler (used by PixArt)
6. **T5XXLEncoder** — First shared encoder (used by PixArt)
7. **PixArt integration** — First model plugin, proves the system end-to-end
8. **SwiftVinetas integration** — PixArtEngine via Pipeline; Flux2Engine unchanged
9. **SwiftVoxAlta integration** — Adopt infrastructure services
10. **FlowMatchEulerScheduler** — Completes catalog (for future model plugins)
11. **AudioRenderer** — Enables future audio diffusion models

---

## Success Criteria

The architecture is working when:

1. **Adding a new diffusion model** requires writing only: backbone (~300 lines), config (~30 lines), key mapping (~50 lines), recipe (~20 lines). Everything else is catalog components and infrastructure.

2. **Shared components are tested once** and validated for every model that uses them. Fixing a bug in SDXLVAEDecoder fixes it for PixArt AND any future model using that VAE.

3. **Pipe segments are independently testable** with synthetic inputs. No full pipeline required to validate a single component.

4. **Pipeline assembly validates compatibility** at construction time, not at generation time. Mismatched components produce compile-time or immediate runtime errors, not silent corruption.

5. **Memory management is centralized**. One MemoryManager coordinates all loaded components. Two-phase loading works identically across all models without per-model memory code.
