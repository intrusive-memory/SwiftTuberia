# SwiftTuberia

**v0.1.2** — Componentized generation pipeline for MLX inference.

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
    .package(url: "https://github.com/intrusive-memory/SwiftTuberia.git", from: "0.1.2")
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

## Documentation

- [AGENTS.md](AGENTS.md) — Architecture and API documentation
- [REQUIREMENTS.md](REQUIREMENTS.md) — Full specification
- [GENERATION_PATHS.md](GENERATION_PATHS.md) — Generation path analysis

## License

MIT License. See [LICENSE](LICENSE) for details.
