# SwiftTubería — Shared Component Catalog

**Parent**: [`REQUIREMENTS.md`](../REQUIREMENTS.md) — SwiftTubería Overview
**Scope**: Concrete implementations of pipe segments that are reused across multiple models. These live in the `TuberíaCatalog` target. This document contains both the component specifications and the canonical Swift configuration type definitions.

---

## Encoders

| Component | Embedding Dim | Used By | Size (int4) |
|---|---|---|---|
| `T5XXLEncoder` | 4096 | PixArt-Sigma, future SD3 | ~1.2 GB |
| `CLIPEncoder` | 768 | Future SDXL, SD models | ~400 MB |

Encoders that are model-specific (Qwen3 for FLUX Klein, Mistral for FLUX Dev) remain in their model plugin packages but conform to the same `TextEncoder` protocol (see [PROTOCOLS.md](PROTOCOLS.md)).

**T5XXLEncoder tokenizer**: The tokenizer is **bundled** with the encoder weight component (same HuggingFace repo, same Acervo component). The encoder's `init(configuration:)` loads the tokenizer files (`tokenizer.json`, `tokenizer_config.json`) via `withComponentAccess` using the same component ID as the weights. It uses `swift-transformers`' `AutoTokenizer.from(modelFolder:)` pointed at the component handle's directory. No separate tokenizer component ID is needed.

---

## Schedulers

| Component | Algorithm | Used By |
|---|---|---|
| `DPMSolverScheduler` | DPM-Solver++ multistep | PixArt, SD, SDXL |
| `FlowMatchEulerScheduler` | Flow matching (rectified flow) | FLUX.2 |
| `DDPMScheduler` | Denoising diffusion | Fallback / training |

---

## Decoders

| Component | Latent Channels | Used By | Size |
|---|---|---|---|
| `SDXLVAEDecoder` | 4, scale 0.13025 | PixArt, SDXL, SD | ~160 MB |

Model-specific decoders (FLUX VAE) remain in their plugin packages.

---

## Renderers

| Component | Input | Output | Stateless |
|---|---|---|---|
| `ImageRenderer` | [B, H, W, 3] float | CGImage | Yes |
| `AudioRenderer` | [B, samples] float | Data (WAV/M4A) | Yes |

Renderers have no model weights. They are pure data transformations.

---

## Component Configuration Types

Each catalog component defines a `Configuration` struct used by pipeline recipes to instantiate the component via `init(configuration:)`. These are the authoritative definitions — model plugin recipes provide values for these fields.

### T5XXLEncoderConfiguration

```swift
public struct T5XXLEncoderConfiguration: Sendable {
    /// Acervo component ID for weights AND tokenizer files (bundled together).
    public let componentId: String              // e.g., "t5-xxl-encoder-int4"
    /// Maximum sequence length for tokenization.
    public let maxSequenceLength: Int           // default: 120 for PixArt, 77 for SD
    /// Embedding dimension (informational — fixed at 4096 for T5-XXL).
    public let embeddingDim: Int                // 4096
}
```

### DPMSolverSchedulerConfiguration

```swift
public struct DPMSolverSchedulerConfiguration: Sendable {
    /// Beta schedule defining the noise schedule.
    public let betaSchedule: BetaSchedule       // e.g., .linear(betaStart: 0.0001, betaEnd: 0.02)
    /// What the model predicts.
    public let predictionType: PredictionType   // e.g., .epsilon
    /// Solver order (1 = first-order Euler, 2 = second-order midpoint).
    public let solverOrder: Int                 // default: 2
    /// Total training timesteps (for beta schedule computation).
    public let trainTimesteps: Int              // default: 1000
}
```

### FlowMatchEulerSchedulerConfiguration

```swift
public struct FlowMatchEulerSchedulerConfiguration: Sendable {
    /// Shift parameter for the sigma schedule.
    public let shift: Float                     // e.g., 1.0 for FLUX
}
```

### SDXLVAEDecoderConfiguration

```swift
public struct SDXLVAEDecoderConfiguration: Sendable {
    /// Acervo component ID for VAE weights.
    public let componentId: String              // e.g., "sdxl-vae-decoder-fp16"
    /// Number of latent channels the decoder expects.
    public let latentChannels: Int              // 4
    /// VAE latent scaling factor (applied internally by the decoder).
    public let scalingFactor: Float             // 0.13025
}
```

### Renderer Configurations

- **ImageRenderer.Configuration**: `Void` — stateless, no configuration needed.
- **AudioRenderer.Configuration**: `Void` — stateless, no configuration needed.

---

## Catalog Component Acervo Descriptors

TuberíaCatalog self-registers the following `ComponentDescriptor` entries at import time. These are the **authoritative** definitions. Model plugins that also register the same component IDs will be silently deduplicated by Acervo (same ID + same repo = no-op).

| Acervo ID | Type | HuggingFace Repo | Key Files | Est. Size |
|---|---|---|---|---|
| `t5-xxl-encoder-int4` | encoder | `intrusive-memory/t5-xxl-int4-mlx` | `*.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `config.json` | ~1.2 GB |
| `sdxl-vae-decoder-fp16` | decoder | `intrusive-memory/sdxl-vae-fp16-mlx` | `*.safetensors`, `config.json` | ~160 MB |

**Repo naming convention**: Shared catalog components use `intrusive-memory/{model}-{quantization}-mlx`. These HuggingFace repos are created during weight conversion (pixart-swift-mlx P7 or a shared conversion step) and populated with MLX safetensors, tokenizer files, and a `config.json` marker.

**DPM-Solver++ and FlowMatchEuler** have no weights — they are pure math. No Acervo component needed.

**ImageRenderer and AudioRenderer** have no weights. No Acervo component needed.

**SHA-256 checksums**: Populated after weight conversion produces the final artifacts. Initial registration may use `sha256: nil` (skip verification) until checksums are computed and backfilled.

---

## Canonical Swift Definitions

These are the **authoritative** configuration type definitions. If any prose above differs from the code below, **this code governs**.

These types live in the `TuberíaCatalog` target. Model plugin recipes reference them.

```swift
// MARK: - T5XXLEncoder Configuration

public struct T5XXLEncoderConfiguration: Sendable {
    /// Acervo component ID for weights AND tokenizer (bundled together).
    public let componentId: String
    /// Maximum sequence length for tokenization.
    public let maxSequenceLength: Int
    /// Embedding dimension (informational — always 4096 for T5-XXL).
    public let embeddingDim: Int

    public init(componentId: String = "t5-xxl-encoder-int4",
                maxSequenceLength: Int = 120,
                embeddingDim: Int = 4096) {
        self.componentId = componentId
        self.maxSequenceLength = maxSequenceLength
        self.embeddingDim = embeddingDim
    }
}

// MARK: - DPMSolverScheduler Configuration

public struct DPMSolverSchedulerConfiguration: Sendable {
    /// Beta schedule defining the noise schedule.
    public let betaSchedule: BetaSchedule
    /// What the model predicts.
    public let predictionType: PredictionType
    /// Solver order (1 = first-order Euler, 2 = second-order midpoint).
    public let solverOrder: Int
    /// Total training timesteps (for beta schedule computation).
    public let trainTimesteps: Int

    public init(betaSchedule: BetaSchedule = .linear(betaStart: 0.0001, betaEnd: 0.02),
                predictionType: PredictionType = .epsilon,
                solverOrder: Int = 2,
                trainTimesteps: Int = 1000) {
        self.betaSchedule = betaSchedule
        self.predictionType = predictionType
        self.solverOrder = solverOrder
        self.trainTimesteps = trainTimesteps
    }
}

// MARK: - FlowMatchEulerScheduler Configuration

public struct FlowMatchEulerSchedulerConfiguration: Sendable {
    /// Shift parameter for the sigma schedule.
    public let shift: Float

    public init(shift: Float = 1.0) {
        self.shift = shift
    }
}

// MARK: - SDXLVAEDecoder Configuration

public struct SDXLVAEDecoderConfiguration: Sendable {
    /// Acervo component ID for VAE weights.
    public let componentId: String
    /// Number of latent channels the decoder expects.
    public let latentChannels: Int
    /// VAE latent scaling factor (applied internally by the decoder).
    public let scalingFactor: Float

    public init(componentId: String = "sdxl-vae-decoder-fp16",
                latentChannels: Int = 4,
                scalingFactor: Float = 0.13025) {
        self.componentId = componentId
        self.latentChannels = latentChannels
        self.scalingFactor = scalingFactor
    }
}

// MARK: - Renderer Configurations
// ImageRenderer.Configuration = Void (stateless, no configuration needed)
// AudioRenderer.Configuration = Void (stateless, no configuration needed)
```
