# SwiftTubería — Pipeline Architecture

**Companion to**: [`../requirements/PIPELINE.md`](../requirements/PIPELINE.md)
**Target**: `Tubería`

---

## Pipeline Type Hierarchy

```swift
protocol GenerationPipeline: Sendable {
    associatedtype Request
    associatedtype Result
    func generate(request: Request, progress: (PipelineProgress) -> Void) async throws -> Result
    func loadModels(progress: (Double, String) -> Void) async throws
    func unloadModels() async
    var memoryRequirement: MemoryRequirement { get }
    var isLoaded: Bool { get }
}

actor DiffusionPipeline<E: TextEncoder, S: Scheduler, B: Backbone, D: Decoder, R: Renderer>
    : GenerationPipeline
    where Request == DiffusionGenerationRequest, Result == DiffusionGenerationResult
```

---

## Request/Result Types

### DiffusionGenerationRequest (authored by consumers)

| Field | Type | Source |
|---|---|---|
| `prompt` | `String` | SwiftVinetas after style composition |
| `negativePrompt` | `String?` | Optional, nil if no CFG negative |
| `width`, `height` | `Int` | From AspectRatio resolution |
| `steps` | `Int` | Model default or user override |
| `guidanceScale` | `Float` | Model default or user override |
| `seed` | `UInt32?` | nil = random |
| `loRA` | `LoRAConfig?` | Engine maps path + scale |
| `referenceImages` | `[CGImage]?` | For img2img |
| `strength` | `Float?` | Denoising strength for img2img |

### DiffusionGenerationResult (returned to consumers)

| Field | Type | Consumed By |
|---|---|---|
| `output` | `RenderedOutput` | Engine extracts CGImage/AudioData |
| `seed` | `UInt32` | Actual seed (important when request seed is nil) |
| `steps` | `Int` | For metadata |
| `guidanceScale` | `Float` | For metadata |
| `duration` | `TimeInterval` | For performance reporting |

### LoRAConfig

```swift
struct LoRAConfig {
    let componentId: String?        // Acervo ID (preferred, takes precedence)
    let localPath: String?          // Fallback (ignored if componentId is non-nil)
    let scale: Float                // 0.0–1.0
    let activationKeyword: String?  // Prepended to prompt
    // Precondition: at least one of componentId or localPath must be non-nil
}
```

---

## PipelineRecipe Protocol

```swift
protocol PipelineRecipe: Sendable {
    associatedtype Encoder: TextEncoder
    associatedtype Sched: Scheduler
    associatedtype Back: Backbone
    associatedtype Dec: Decoder
    associatedtype Rend: Renderer

    var encoderConfig:   Encoder.Configuration { get }
    var schedulerConfig: Sched.Configuration   { get }
    var backboneConfig:  Back.Configuration    { get }
    var decoderConfig:   Dec.Configuration     { get }
    var rendererConfig:  Rend.Configuration    { get }

    var supportsImageToImage: Bool { get }
    var unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy { get }
    var allComponentIds: [String] { get }

    func quantizationFor(_ role: PipelineRole) -> QuantizationConfig
    func validate() throws
}
```

### Ecosystem Recipes

| Recipe | Encoder | Scheduler | Backbone | Decoder | Renderer |
|---|---|---|---|---|---|
| `PixArtRecipe` | T5XXLEncoder | DPMSolverScheduler | PixArtDiT | SDXLVAEDecoder | ImageRenderer |
| FLUX.2 Klein (future) | Qwen3TextEncoder | FlowMatchEulerScheduler | FluxDiT | FluxVAEDecoder | ImageRenderer |
| FLUX.2 Dev (future) | MistralTextEncoder | FlowMatchEulerScheduler | FluxDiT | FluxVAEDecoder | ImageRenderer |

---

## Orchestration Flow

```
1. E.encode(text, maxLength) ──────▶ TextEncoderOutput {embeddings, mask}
                                           │
2. Initial latents:                        │
   - txt2img: random noise                 │
   - img2img: BidirectionalDecoder.encode  │
     + Scheduler.addNoise                  │
                                           │
3. S.configure(steps, startTimestep) ──▶ SchedulerPlan {timesteps, sigmas}
                                           │
4. Denoising loop:                         │
   for each timestep in plan:              │
     ├─ Build BackboneInput ◀──────────────┘
     │    {latents, conditioning=embeddings, conditioningMask=mask, timestep}
     ├─ B.forward(input) ─────────▶ noise prediction
     ├─ CFG: uncond + scale*(cond-uncond)
     └─ S.step(output, timestep, sample) ──▶ updated latents

5. D.decode(final_latents) ──────▶ DecodedOutput {data, metadata}

6. R.render(decoded) ────────────▶ RenderedOutput
```

---

## CFG Strategy

```swift
enum UnconditionalEmbeddingStrategy {
    case emptyPrompt              // PixArt, SD: encode "" through same encoder
    case zeroVector(shape: [Int]) // All-zero embedding
    case none                     // FLUX: guidance embedded in model, no CFG
}
```

---

## Two-Phase Loading

```
Phase 1 — Conditioning:
  Load TextEncoder → encode prompt → unload TextEncoder → clearGPUCache()

Phase 2 — Generation:
  Load Backbone + Decoder → denoise → decode → unload
```

`MemoryRequirement { peakMemoryBytes, phasedMemoryBytes }` — consumers choose strategy based on device memory.

---

## Assembly Validation (6 checks)

1. **Completeness**: All components provided
2. **Encoder→Backbone (dim)**: `encoder.outputEmbeddingDim == backbone.expectedConditioningDim`
3. **Encoder→Backbone (seq)**: `encoder.maxSequenceLength == backbone.expectedMaxSequenceLength`
4. **Backbone→Decoder**: `backbone.outputLatentChannels == decoder.expectedInputChannels`
5. **Decoder→Renderer**: Output modality matches renderer type
6. **Img2Img**: If `supportsImageToImage`, Decoder conforms to `BidirectionalDecoder`

---

## Error Model

```swift
enum PipelineError: Error {
    // Assembly
    case incompatibleComponents(inlet: String, outlet: String, reason: String)
    case missingComponent(role: String)
    // Infrastructure
    case modelNotDownloaded(component: String)
    case insufficientMemory(required: UInt64, available: UInt64, component: String)
    case weightLoadingFailed(component: String, reason: String)
    case downloadFailed(component: String, reason: String)
    // Generation
    case encodingFailed(reason: String)
    case generationFailed(step: Int, reason: String)
    case decodingFailed(reason: String)
    case renderingFailed(reason: String)
    // Control
    case cancelled
}
```

---

## Progress Reporting

```swift
enum PipelineProgress {
    case downloading(component: String, fraction: Double)
    case loading(component: String, fraction: Double)
    case encoding(fraction: Double)
    case generating(step: Int, totalSteps: Int, elapsed: TimeInterval)
    case decoding
    case rendering
    case complete(duration: TimeInterval)
}
```
