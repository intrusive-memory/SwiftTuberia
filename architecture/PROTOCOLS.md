# SwiftTubería — Protocols Architecture

**Companion to**: [`../requirements/PROTOCOLS.md`](../requirements/PROTOCOLS.md)
**Target**: `Tubería`

---

## Protocol Hierarchy

```
                    WeightedSegment
                   ╱       │       ╲
          TextEncoder   Backbone   Decoder ──▶ BidirectionalDecoder
                                                  (optional extension)

          Scheduler (no weights)
          Renderer  (no weights, stateless)
```

---

## WeightedSegment Lifecycle

All weight-bearing segments share this contract:

```swift
protocol WeightedSegment: Sendable {
    func apply(weights: ModuleParameters) throws    // Receive loaded params
    func unload()                                    // Release params
    var estimatedMemoryBytes: Int { get }            // Static estimate
    var isLoaded: Bool { get }                       // Current state
    var keyMapping: KeyMapping { get }               // safetensors key → module key
    var tensorTransform: TensorTransform? { get }    // Optional per-tensor transform (default: nil)
}
```

**Conformance pattern**: `final class` with `@unchecked Sendable`. DiffusionPipeline actor serializes access.

**Loading sequence** (orchestrated by pipeline, not by segment):
1. `Acervo.ensureComponentReady(id)` — download
2. `WeightLoader.load(componentId:, keyMapping:, tensorTransform:, quantization:)` → `ModuleParameters`
3. `segment.apply(weights:)` — segment assigns params to internal modules
4. `MemoryManager.registerLoaded(component:, bytes:)`

---

## Pipe Segment Contracts

### TextEncoder

```
inlet:  TextEncoderInput  { text: String, maxLength: Int }
outlet: TextEncoderOutput { embeddings: MLXArray [B, seq, dim], mask: MLXArray [B, seq] }
```

| Property | Type | Validated Against |
|---|---|---|
| `outputEmbeddingDim` | `Int` | `Backbone.expectedConditioningDim` |
| `maxSequenceLength` | `Int` | `Backbone.expectedMaxSequenceLength` |

**Ecosystem instances**: T5XXLEncoder (catalog, dim=4096, seq=120/77), CLIPEncoder (catalog, dim=768), Qwen3TextEncoder (flux-2-swift-mlx), MistralTextEncoder (flux-2-swift-mlx)

### Scheduler

```
inlet:  steps: Int, startTimestep: Int?
outlet: SchedulerPlan { timesteps: [Int], sigmas: [Float] }

per-step:
  inlet:  output: MLXArray, timestep: Int, sample: MLXArray
  outlet: MLXArray (denoised sample)
```

**No weights. No WeightedSegment conformance.**

Methods: `configure(steps:startTimestep:)`, `step(output:timestep:sample:)`, `addNoise(to:noise:at:)`, `reset()`

**Ecosystem instances**: DPMSolverScheduler (catalog), FlowMatchEulerScheduler (catalog), DDPMScheduler (catalog)

### Backbone

```
inlet:  BackboneInput {
            latents:          MLXArray [B, H/8, W/8, channels]
            conditioning:     MLXArray [B, seq, dim]
            conditioningMask: MLXArray [B, seq]
            timestep:         MLXArray (scalar or [B])
        }
outlet: MLXArray [B, H/8, W/8, channels]  (noise prediction)
```

| Property | Type | Validated Against |
|---|---|---|
| `expectedConditioningDim` | `Int` | `TextEncoder.outputEmbeddingDim` |
| `outputLatentChannels` | `Int` | `Decoder.expectedInputChannels` |
| `expectedMaxSequenceLength` | `Int` | `TextEncoder.maxSequenceLength` |

**Ecosystem instances**: PixArtDiT (pixart-swift-mlx), FluxDiT (flux-2-swift-mlx, future)

### Decoder

```
inlet:  MLXArray (latents from final denoising step)
outlet: DecodedOutput { data: MLXArray, metadata: DecoderMetadata }
```

| Property | Type | Validated Against |
|---|---|---|
| `expectedInputChannels` | `Int` | `Backbone.outputLatentChannels` |
| `scalingFactor` | `Float` | (internal use only) |

Scaling (`latents * (1.0 / scalingFactor)`) applied **internally by decoder**, not by pipeline.

**Ecosystem instances**: SDXLVAEDecoder (catalog, channels=4, scale=0.13025), FluxVAEDecoder (flux-2-swift-mlx, future)

### BidirectionalDecoder (extends Decoder)

Adds `encode(_ pixels: MLXArray) throws -> MLXArray` for img2img support.

### Renderer

```
inlet:  DecodedOutput
outlet: RenderedOutput (.image(CGImage) | .audio(AudioData) | .video(VideoFrames))
```

**No weights. Stateless. Configuration = Void for image/audio.**

**Ecosystem instances**: ImageRenderer (catalog), AudioRenderer (catalog)

---

## Field Mapping (Pipeline Internal)

```
TextEncoderOutput.embeddings  →  BackboneInput.conditioning
TextEncoderOutput.mask        →  BackboneInput.conditioningMask
```

This mapping is performed by `DiffusionPipeline` during orchestration. Neither encoder nor backbone knows the other's field names.

---

## Supporting Types

```swift
struct ModuleParameters { let parameters: [String: MLXArray] }
typealias KeyMapping = @Sendable (String) -> String?
typealias TensorTransform = @Sendable (String, MLXArray) -> MLXArray
enum QuantizationConfig { case asStored, float16, bfloat16, int4(groupSize:), int8(groupSize:) }
```
