# SwiftTubería — Catalog Architecture

**Companion to**: [`../requirements/CATALOG.md`](../requirements/CATALOG.md)
**Target**: `TuberíaCatalog`

---

## Catalog Components

### Encoders

| Component | Protocol | outputEmbeddingDim | Compatible Backbones | Acervo ID |
|---|---|---|---|---|
| `T5XXLEncoder` | `TextEncoder` | 4096 | PixArtDiT | `t5-xxl-encoder-int4` |
| `CLIPEncoder` (future) | `TextEncoder` | 768 | SD, SDXL backbones | TBD |

### Schedulers (no weights, no Acervo component)

| Component | Protocol | Algorithm | Compatible Models |
|---|---|---|---|
| `DPMSolverScheduler` | `Scheduler` | DPM-Solver++ multistep | PixArt, SD, SDXL |
| `FlowMatchEulerScheduler` | `Scheduler` | Rectified flow | FLUX.2 |
| `DDPMScheduler` | `Scheduler` | DDPM | Fallback/training |

### Decoders

| Component | Protocol | expectedInputChannels | scalingFactor | Acervo ID |
|---|---|---|---|---|
| `SDXLVAEDecoder` | `Decoder` | 4 | 0.13025 | `sdxl-vae-decoder-fp16` |

### Renderers (no weights, no Acervo component)

| Component | Protocol | Input Shape | Output |
|---|---|---|---|
| `ImageRenderer` | `Renderer` | [B, H, W, 3] float | `.image(CGImage)` |
| `AudioRenderer` | `Renderer` | [B, samples] float | `.audio(AudioData)` |

---

## Configuration Types (consumed by PipelineRecipe)

### T5XXLEncoderConfiguration

```swift
struct T5XXLEncoderConfiguration: Sendable {
    let componentId: String          // Default: "t5-xxl-encoder-int4"
    let maxSequenceLength: Int       // PixArt: 120, SD: 77
    let embeddingDim: Int            // Always 4096 for T5-XXL
}
```

**Tokenizer**: Bundled in same Acervo component as weights. Loaded by encoder during `init(configuration:)` via `withComponentAccess`.

### DPMSolverSchedulerConfiguration

```swift
struct DPMSolverSchedulerConfiguration: Sendable {
    let betaSchedule: BetaSchedule   // .linear(betaStart: 0.0001, betaEnd: 0.02)
    let predictionType: PredictionType // .epsilon
    let solverOrder: Int             // 2
    let trainTimesteps: Int          // 1000
}
```

### FlowMatchEulerSchedulerConfiguration

```swift
struct FlowMatchEulerSchedulerConfiguration: Sendable {
    let shift: Float                 // 1.0 for FLUX
}
```

### SDXLVAEDecoderConfiguration

```swift
struct SDXLVAEDecoderConfiguration: Sendable {
    let componentId: String          // Default: "sdxl-vae-decoder-fp16"
    let latentChannels: Int          // 4
    let scalingFactor: Float         // 0.13025
}
```

### Renderer Configurations

`ImageRenderer.Configuration = Void`
`AudioRenderer.Configuration = Void`

---

## Acervo Self-Registration

TuberíaCatalog registers shared component descriptors at import time:

| Acervo ID | Type | HuggingFace Repo | Key Files |
|---|---|---|---|
| `t5-xxl-encoder-int4` | .encoder | `intrusive-memory/t5-xxl-int4-mlx` | `*.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `config.json` |
| `sdxl-vae-decoder-fp16` | .decoder | `intrusive-memory/sdxl-vae-fp16-mlx` | `*.safetensors`, `config.json` |

Model plugins that also register these IDs are silently deduplicated by Acervo.

---

## Shape Contract Validation Points

```
T5XXLEncoder.outputEmbeddingDim (4096)
    ↕  must equal
PixArtDiT.expectedConditioningDim (4096)

T5XXLEncoder.maxSequenceLength (120)
    ↕  must equal
PixArtDiT.expectedMaxSequenceLength (120)

PixArtDiT.outputLatentChannels (4)
    ↕  must equal
SDXLVAEDecoder.expectedInputChannels (4)

SDXLVAEDecoder output: [B, H, W, 3] float pixels
    ↕  compatible with
ImageRenderer input: [B, H, W, 3] float → CGImage
```
