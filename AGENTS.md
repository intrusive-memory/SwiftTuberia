# AGENTS.md

This file provides comprehensive documentation for AI agents working with the SwiftTuberia codebase.

**Version**: 0.2.7

---

## Recent Changes

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
xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'
```

Available schemes: `SwiftTuberia-Package`, `Tuberia`, `TuberiaCatalog`. Run `xcodebuild -list` to confirm. **Never use `-scheme SwiftTuberia`** — that scheme does not exist.

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
