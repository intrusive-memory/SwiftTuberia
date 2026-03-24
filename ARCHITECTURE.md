# SwiftTuberia — Architecture (Ecosystem Interface Reference)

**Companion to**: [`REQUIREMENTS.md`](REQUIREMENTS.md)
**Role in ecosystem**: Central pipeline system. Provides protocols, infrastructure, and shared components.

---

## Dependency Position

```
pixart-swift-mlx ──▶ Tuberia + TuberiaCatalog
SwiftVinetas ──────▶ Tuberia + TuberiaCatalog
SwiftVoxAlta ──────▶ Tuberia only (infrastructure)
Tuberia ───────────▶ SwiftAcervo, mlx-swift, swift-transformers
TuberiaCatalog ────▶ Tuberia
```

---

## Two Products

| Product | Target | Contains | Imported By |
|---|---|---|---|
| `Tuberia` | Protocols + Infrastructure | Pipe segment protocols, WeightLoader, MemoryManager, DeviceCapability, DiffusionPipeline, PipelineRecipe, LoRA, Progress, Errors | All consumers |
| `TuberiaCatalog` | Shared Components | T5XXLEncoder, SDXLVAEDecoder, DPMSolverScheduler, FlowMatchEulerScheduler, ImageRenderer, AudioRenderer + their Configuration types | Model plugins, SwiftVinetas |

**SwiftVoxAlta imports only `Tuberia`** — it uses MemoryManager and DeviceCapability but no diffusion components.

---

## Sub-Architecture Documents

| Document | Scope |
|---|---|
| [`architecture/PROTOCOLS.md`](architecture/PROTOCOLS.md) | Pipe segment protocols and WeightedSegment lifecycle |
| [`architecture/PIPELINE.md`](architecture/PIPELINE.md) | DiffusionPipeline, PipelineRecipe, request/result types |
| [`architecture/CATALOG.md`](architecture/CATALOG.md) | Shared component configurations and Acervo descriptors |
| [`architecture/INFRASTRUCTURE.md`](architecture/INFRASTRUCTURE.md) | WeightLoader, MemoryManager, DeviceCapability |

---

## Key Exported Types Summary

### From `Tuberia` target

**Protocols**: `TextEncoder`, `Scheduler`, `Backbone`, `Decoder`, `Renderer`, `BidirectionalDecoder`, `WeightedSegment`, `GenerationPipeline`, `PipelineRecipe`

**Pipeline types**: `DiffusionPipeline<E,S,B,D,R>`, `DiffusionGenerationRequest`, `DiffusionGenerationResult`, `LoRAConfig`, `MemoryRequirement`

**Infrastructure**: `WeightLoader`, `MemoryManager`, `DeviceCapability`

**Data types**: `BackboneInput`, `TextEncoderInput`, `TextEncoderOutput`, `DecodedOutput`, `RenderedOutput`, `SchedulerPlan`, `ModuleParameters`, `PipelineProgress`, `PipelineError`

**Enums**: `QuantizationConfig`, `BetaSchedule`, `PredictionType`, `UnconditionalEmbeddingStrategy`, `PipelineRole`, `ComponentType` (re-exported from Acervo)

### From `TuberiaCatalog` target

**Components**: `T5XXLEncoder`, `SDXLVAEDecoder`, `DPMSolverScheduler`, `FlowMatchEulerScheduler`, `ImageRenderer`, `AudioRenderer`

**Configurations**: `T5XXLEncoderConfiguration`, `SDXLVAEDecoderConfiguration`, `DPMSolverSchedulerConfiguration`, `FlowMatchEulerSchedulerConfiguration`

---

## Data Flow Through the System

```
Model Plugin                    SwiftTuberia                         SwiftAcervo
─────────────                   ────────────                         ───────────

PipelineRecipe ──────────────▶ DiffusionPipeline.init(recipe:)
                                  │
                                  ├─ Instantiates components via init(configuration:)
                                  ├─ Validates shape contracts
                                  │
                               loadModels()
                                  │
                                  ├─ Acervo.ensureComponentsReady() ──▶ download if needed
                                  │
                                  ├─ WeightLoader.load() ─────────────▶ withComponentAccess()
                                  │     │                                    │
                                  │     ├─ keyMapping (from segment)          ├─ handle.urls()
                                  │     ├─ tensorTransform (from segment)     └─ scoped file access
                                  │     └─ quantization (from recipe)
                                  │
                                  ├─ segment.apply(weights:)
                                  └─ MemoryManager.registerLoaded()

                               generate(request:)
                                  │
                                  ├─ TextEncoder.encode() → TextEncoderOutput
                                  ├─ Scheduler.configure() → SchedulerPlan
                                  ├─ for each timestep:
                                  │     ├─ Backbone.forward(BackboneInput) → MLXArray
                                  │     └─ Scheduler.step() → updated latents
                                  ├─ Decoder.decode() → DecodedOutput
                                  └─ Renderer.render() → RenderedOutput
```

---

## Assembly-Time Validation Points

| Check | Left Side | Right Side | Error |
|---|---|---|---|
| Encoder→Backbone (dim) | `TextEncoder.outputEmbeddingDim` | `Backbone.expectedConditioningDim` | `incompatibleComponents` |
| Encoder→Backbone (seq) | `TextEncoder.maxSequenceLength` | `Backbone.expectedMaxSequenceLength` | `incompatibleComponents` |
| Backbone→Decoder | `Backbone.outputLatentChannels` | `Decoder.expectedInputChannels` | `incompatibleComponents` |
| Img2Img support | `recipe.supportsImageToImage` | `Decoder is BidirectionalDecoder` | `incompatibleComponents` |
| Component IDs | `recipe.allComponentIds` | `Acervo.isComponentReady()` | `modelNotDownloaded` |
