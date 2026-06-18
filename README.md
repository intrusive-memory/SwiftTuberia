# SwiftTuberia

**v0.7.5** — Componentized generation pipeline for MLX inference.

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
    .package(url: "https://github.com/intrusive-memory/SwiftTuberia.git", from: "0.7.4")
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

## App Group configuration (required)

This package depends on [SwiftAcervo](https://github.com/intrusive-memory/SwiftAcervo) for shared model storage. SwiftAcervo v0.10.0 resolves its App Group ID in this order: `ACERVO_APP_GROUP_ID` env var → `com.apple.security.application-groups` entitlement (macOS only) → `fatalError`. There is **no silent fallback**.

- **Signed UI apps (macOS / iOS)**: declare `com.apple.security.application-groups` with `group.intrusive-memory.models` in your `.entitlements` file. iOS apps additionally need `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the launch environment.
- **CLI tools, scripts, CI jobs, test runners**: export `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the shell or job environment. The standard place is `~/.zprofile`:

    ```sh
    export ACERVO_APP_GROUP_ID=group.intrusive-memory.models
    ```

Without this, `Acervo.sharedModelsDirectory` traps with `fatalError`. See [SwiftAcervo's USAGE.md](https://github.com/intrusive-memory/SwiftAcervo/blob/main/USAGE.md) for full details.

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

## Telemetry

Tuberia emits typed `TuberiaTelemetryEvent` events covering 27 cases across the
full diffusion pipeline — lifecycle, assembly validation, memory gating, weight
loading, LoRA, text-encoder, scheduler, per-step denoise, CFG cast, backbone,
decoder, renderer, numerical anomalies, and thrown errors. Hosts subscribe by
conforming to `TuberiaTelemetryReporter` and installing via
`TuberiaTelemetry.setReporter(_:)`.

```swift
import Tuberia

struct MyTelemetryReporter: TuberiaTelemetryReporter {
    func capture(_ event: TuberiaTelemetryEvent) async {
        print("[tuberia] \(event)")
    }
}

// At process startup:
TuberiaTelemetry.setReporter(MyTelemetryReporter())

// At shutdown:
TuberiaTelemetry.setReporter(nil)
```

For test isolation, `DiffusionPipeline.setTelemetry(_:)` installs a reporter on a
single pipeline instance; an instance reporter always takes priority over the
process-wide one.

## Documentation

- [AGENTS.md](AGENTS.md) — Architecture, API, and interop documentation
- [REQUIREMENTS.md](REQUIREMENTS.md) — Full specification
- [GENERATION_PATHS.md](docs/GENERATION_PATHS.md) — Generation path analysis

## License

MIT License. See [LICENSE](LICENSE) for details.
