# AGENTS.md

This file provides comprehensive documentation for AI agents working with the SwiftTuberia codebase.

**Version**: 0.2.0

---

## Project Overview

SwiftTuberia ("tuberia" -- Spanish for "plumbing" or "piping system") is a componentized generation pipeline for MLX inference. It provides typed pipe segment protocols, a diffusion pipeline compositor, shared component catalog, and infrastructure services.

**Key concept**: Each pipeline component is a typed pipe segment with a defined inlet and outlet. Model-specific packages provide only their unique backbone architecture and a recipe declaring which pipes to connect. Everything else comes from the shared catalog.

## Architecture

Two products:
- **Tuberia** -- Protocols + infrastructure (TextEncoder, Scheduler, Backbone, Decoder, Renderer protocols; WeightLoader, MemoryManager, LoRA, Progress)
- **TuberiaCatalog** -- Concrete shared components (T5-XXL, SDXL VAE, DPM-Solver++, FlowMatch Euler, ImageRenderer, AudioRenderer)

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
xcodebuild build -scheme SwiftTuberia -destination 'platform=macOS'
xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS'
```

See [REQUIREMENTS.md](REQUIREMENTS.md) for the complete specification.
See [GENERATION_PATHS.md](GENERATION_PATHS.md) for generation path analysis.
