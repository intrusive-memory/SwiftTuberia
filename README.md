# SwiftTuberia

**v0.3.6** — Componentized generation pipeline for MLX inference.

## Overview

SwiftTuberia provides typed pipe segment protocols, a diffusion pipeline compositor, shared component catalog, and infrastructure services for building MLX-based generation pipelines.

### Key Features

- **Pipe Segment Protocols** — TextEncoder, Scheduler, Backbone, Decoder, Renderer with typed inlets/outlets
- **DiffusionPipeline** — Compositor that wires segments together via PipelineRecipe
- **Shared Catalog** — T5-XXL encoder, SDXL VAE decoder, DPM-Solver++, FlowMatch Euler schedulers, Image/Audio renderers
- **Infrastructure** — WeightLoader, MemoryManager, LoRA support, progress tracking

## Requirements

- macOS 26.0+ / iOS 26.0+
- Swift 6.2+
- Xcode 26+
- Apple Silicon (M1+)

## Installation

Add SwiftTuberia to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/intrusive-memory/SwiftTuberia.git", from: "0.3.6")
]
```

Then add the products you need:

```swift
.target(
    name: "YourTarget",
    dependencies: [
        .product(name: "Tuberia", package: "SwiftTuberia"),
        .product(name: "TuberiaCatalog", package: "SwiftTuberia")
    ]
)
```

- **Tuberia** — Protocols + infrastructure (use when building custom components)
- **TuberiaCatalog** — Concrete shared components (includes Tuberia via `@_exported import`)

## LoRA Adapter Compatibility

`LoRALoader` is the shared LoRA merge engine (`W' = W + scale × (B @ A)`) used by all downstream model packages. It normalizes key names across training framework conventions before merging.

### Supported conventions

| Training framework | Key format | Status |
|-------------------|-----------|--------|
| HuggingFace / Ostris | `layer.weight.lora_A` / `.lora_B` | ✅ |
| HuggingFace dot-weight | `layer.lora_A.weight` / `.lora_B.weight` | ✅ |
| Diffusers | `layer.lora_down` / `.lora_up` (with or without `.weight`) | ✅ |
| PEFT (Flux.2, SD3) | `base_model.model.layer.lora_A.weight` | ❌ `base_model.model.` prefix not stripped |
| Older SD/UNet adapters | `unet.layer.lora_A.weight` | ❌ `unet.` prefix not stripped |

### Downstream package interop

| Package | Adapter source | Status |
|---------|---------------|--------|
| [`pixart-swift-mlx`](../pixart-swift-mlx) | HuggingFace-style keys (`layer.lora_A.weight`) | ✅ Fully compatible |
| [`flux-2-swift-mlx`](../flux-2-swift-mlx) | PEFT-trained adapters (`base_model.model.` prefix) | ❌ Prefix not stripped — handled locally in `flux-2-swift-mlx` pending fix here |

**Priority**: `base_model.model.` prefix stripping must be added to `LoRALoader.parseLoRAKey` before any load path in either downstream package can route through the shared loader for PEFT adapters. See `AGENTS.md` § LoRA Adapter Interoperability for full policy and test requirements.

## Documentation

- [AGENTS.md](AGENTS.md) — Architecture, API, and interop documentation
- [REQUIREMENTS.md](REQUIREMENTS.md) — Full specification
- [GENERATION_PATHS.md](GENERATION_PATHS.md) — Generation path analysis

## License

MIT License. See [LICENSE](LICENSE) for details.
