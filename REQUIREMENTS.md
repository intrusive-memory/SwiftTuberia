# SwiftTubería — Requirements

**Status**: DRAFT — debate and refine before implementation.
**Parent project**: [`PROJECT_PIPELINE.md`](../PROJECT_PIPELINE.md) — Unified MLX Inference Architecture (§2. SwiftTubería, Waves 1–2)
**Scope**: Unified, componentized generation stack for MLX inference on Apple Silicon. All model-specific packages (pixart-swift-mlx, flux-2-swift-mlx, future video/audio models) plug into this pipeline rather than building standalone stacks.

---

## Motivation

Every MLX inference workflow follows the same pattern:

```
Condition(prompt) → Generate(latents, condition, timesteps) → Decode(raw output) → Render(final format)
```

Today, each model repo (flux-2-swift-mlx, pixart-swift-mlx, SwiftVoxAlta) rebuilds this entire stack from scratch — model downloading, weight loading, quantization, memory management, scheduling, VAE decoding, image rendering. The model-specific code (the unique neural network architecture) is typically ~20% of each library; the other ~80% is shared infrastructure rebuilt each time.

SwiftTubería inverts this. It provides the pipeline system and shared components. Model packages provide only the delta — their unique backbone architecture, weight key mapping, and a recipe that declares which pipe segments to connect.

### Design Metaphor: Tubería (Plumbing)

*Tubería* is Spanish for "plumbing" or "piping system" — the network of pipes that carries water from source to destination. This is not an analogy. It is a literal description of what this library does.

A building's plumbing system has standardized pipe segments with typed connections. A half-inch copper inlet connects to a half-inch copper outlet — always. You can test a single pipe segment by running water through it without connecting the entire system. You can swap a corroded segment for a new one without re-plumbing the whole building. A plumber assembles a system from a catalog of standard parts, adding custom fittings only where the building demands something unique.

SwiftTubería works the same way:

- **Pipe segments** are typed components (TextEncoder, Scheduler, Backbone, Decoder, Renderer) with defined inlets and outlets
- **Recipes** are blueprints — they declare which pipe segments connect to form a complete system
- **The catalog** is the hardware store — standard components (T5-XXL encoder, SDXL VAE decoder, DPM-Solver scheduler) that any model can pull off the shelf
- **Model plugins** provide only the custom fitting — the unique backbone architecture that is specific to one model. Everything else comes from the catalog.
- **Assembly-time validation** ensures every connection is compatible before anything flows. A 4096-dim encoder outlet connected to a 768-dim backbone inlet fails at assembly, not at generation.

The plumbing metaphor also explains SwiftVoxAlta's relationship: TTS doesn't use the pipe system (it's autoregressive, not diffusion), but it does connect to the **infrastructure** — the water meter (MemoryManager), the pressure regulator (DeviceCapability), and the utility company (SwiftAcervo). You don't need pipes to benefit from shared infrastructure.

```
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│  Text     │───▶│  Backbone  │───▶│  Decoder  │───▶│ Renderer  │
│  Encoder  │    │ (per-model)│    │           │    │           │
│           │    │            │    │           │    │           │
│ inlet:    │    │ inlet:     │    │ inlet:    │    │ inlet:    │
│  String   │    │ Embeddings │    │ MLXArray  │    │ Decoded   │
│ outlet:   │    │ + Latents  │    │ (latents) │    │  Output   │
│ Embeddings│    │ + Timestep │    │ outlet:   │    │ outlet:   │
└──────────┘    │ outlet:    │    │ Decoded   │    │ Final     │
                │  MLXArray  │    │  Output   │    │  Output   │
                └───────────┘    └──────────┘    └──────────┘
```

A diffusion pipeline also includes a **Scheduler** segment that drives the backbone iteratively:

```
┌───────────┐
│ Scheduler  │──── drives iteration ────▶ Backbone
│            │◀─── noise prediction ─────│
│ inlet:     │
│  config    │
│ outlet:    │
│  timesteps │
│  + step()  │
└───────────┘
```

---

## R1. Platforms

```swift
platforms: [.macOS(.v26), .iOS(.v26)]
```

macOS and iPadOS (M-series) are first-class targets. This enables lightweight models (PixArt, future small models) to run on iPad while heavier models (FLUX.2) remain Mac-only — determined by the model's memory requirements, not by the pipeline.

---

## R2. Component Protocols — The Pipe Segments

Each protocol defines a pipe segment with typed inlet/outlet. The boundary contracts are **hard** — a component receives exactly what it declares and produces exactly what it promises. No ambient state, no side channels.

### R2.0 Shared Lifecycle: WeightedSegment

Pipe segments that carry model weights (TextEncoder, Backbone, Decoder) share a common lifecycle. The pipeline loads weights centrally via `WeightLoader` and delivers them to segments through this protocol:

```swift
/// Remapped, quantized parameter tensors ready for module assignment.
public struct ModuleParameters: Sendable {
    public let parameters: [String: MLXArray]
}

/// Key remapping function: safetensors key → module key. Return nil to skip a key.
public typealias KeyMapping = @Sendable (String) -> String?

/// Optional per-tensor transform applied after key remapping, before quantization.
/// Used for layout conversion (e.g., PyTorch Conv2d [O,I,kH,kW] → MLX [O,kH,kW,I]).
public typealias TensorTransform = @Sendable (String, MLXArray) -> MLXArray

/// Quantization strategy for weight loading.
public enum QuantizationConfig: Sendable {
    /// Load weights as stored in the safetensors file. No conversion.
    /// Default for pre-quantized models (the Acervo component ID already encodes
    /// the quantization level, e.g., "pixart-sigma-xl-dit-int4").
    case asStored
    /// Convert to float16 after loading.
    case float16
    /// Convert to bfloat16 after loading.
    case bfloat16
    /// Quantize to 4-bit after loading.
    case int4(groupSize: Int = 64)
    /// Quantize to 8-bit after loading.
    case int8(groupSize: Int = 64)
}
```

```swift
/// Lifecycle for pipe segments that carry model weights.
/// Conformed to by TextEncoder, Backbone, and Decoder.
/// Scheduler and Renderer do NOT conform — they have no weights.
public protocol WeightedSegment: Sendable {
    /// Apply loaded weights to the module graph. Called by the pipeline after
    /// WeightLoader has parsed safetensors, remapped keys, and quantized.
    func apply(weights: ModuleParameters) throws

    /// Release all weight memory. Called by the pipeline between loading phases.
    func unload()

    /// Estimated GPU memory when loaded.
    var estimatedMemoryBytes: Int { get }

    /// Whether weights are currently loaded.
    var isLoaded: Bool { get }

    /// Key remapping for this segment's weight files.
    /// Plugins implement however they prefer — dictionary lookup, regex, prefix stripping.
    var keyMapping: KeyMapping { get }

    /// Optional per-tensor transform (e.g., conv weight transposition).
    /// Default: nil (no transform).
    var tensorTransform: TensorTransform? { get }
}

extension WeightedSegment {
    public var tensorTransform: TensorTransform? { nil }
}
```

**Loading flow** — the pipeline orchestrates, segments receive:
```
Pipeline.loadModels():
  for each weighted segment in recipe:
    1. Acervo.ensureComponentReady(componentId)           // download if needed
    2. weights = WeightLoader.load(                       // parse + remap + quantize
         componentId: componentId,
         keyMapping: segment.keyMapping,
         tensorTransform: segment.tensorTransform,
         quantization: recipe.quantizationFor(segment)
       )
    3. segment.apply(weights: weights)                    // segment receives clean params
    4. MemoryManager.registerLoaded(componentId, bytes)   // track memory
```

No pipe segment ever touches Acervo, URLs, or safetensors parsing. The WeightLoader is the only code that reads safetensors. The pipe segment is the only code that knows its internal module structure.

**Non-weight resources** (tokenizers, configs): Loaded by the segment during initialization, not through `apply(weights:)`. The segment's `Configuration` type includes the Acervo component ID for non-weight resources, and the segment loads them via `withComponentAccess` during `init`. This is an implementation detail — the pipeline doesn't manage tokenizer loading.

### R2.1 TextEncoder

Converts raw text into dense embeddings that condition the generation.

```
inlet:  TextEncoderInput  { text: String, maxLength: Int }
outlet: TextEncoderOutput { embeddings: MLXArray [B, seq, dim], mask: MLXArray [B, seq] }
```

**Lifecycle**: Conforms to `WeightedSegment` (see R2.0). Pipeline loads weights via `WeightLoader` → calls `apply(weights:)`. Non-weight resources (tokenizers) loaded by the encoder during initialization through its `Configuration`.
**Shape contract**: `var outputEmbeddingDim: Int { get }` — e.g., 4096 for T5-XXL, 768 for CLIP. Validated at pipeline assembly against Backbone's `expectedConditioningDim`.

Different models use different text encoders (T5-XXL for PixArt, Qwen3 for FLUX Klein, Mistral for FLUX Dev, CLIP for future SD models). The encoder protocol is the same; the implementation varies.

### R2.2 Scheduler

Drives the iterative denoising loop for diffusion models. Stateful within a single generation, stateless between generations.

```
inlet:  SchedulerConfig   { steps: Int, guidanceScale: Float, betaSchedule: BetaSchedule?, predictionType: PredictionType }
outlet: SchedulerPlan     { timesteps: [Int], sigmas: [Float] }

per-step:
  inlet:  SchedulerStepInput  { modelOutput: MLXArray, timestep: Int, sample: MLXArray }
  outlet: MLXArray             (denoised sample)
```

```swift
public enum BetaSchedule {
    case linear(betaStart: Float, betaEnd: Float)
    case cosine
    case sqrt
}
```

`betaSchedule` is optional — DDPM-family schedulers (DPM-Solver++, DDPM) require it; flow-matching schedulers (FlowMatchEuler) ignore it and use sigma schedules instead.

```swift
public enum PredictionType: String, Sendable {
    case epsilon    // predict noise (PixArt, SD, SDXL)
    case velocity   // predict velocity / v-prediction (FLUX)
    case sample     // predict clean sample directly
}
```

**Methods**:
- `configure(steps:) -> SchedulerPlan` — compute timestep schedule
- `step(output:timestep:sample:) -> MLXArray` — single denoising step
- `addNoise(to:noise:at:) -> MLXArray` — for img2img initialization
- `reset()` — clear state between generations

**Timestep shape**: Schedulers produce scalar timesteps. The DiffusionPipeline normalizes to scalar before calling `backbone.forward()`. Backbones MUST accept both scalar and `[B]` shapes defensively.

### R2.3 Backbone

The model-specific neural network — the ONLY component that model plugin packages must implement from scratch. Everything else is either shared or selected from the catalog.

```
inlet:  BackboneInput {
            latents: MLXArray,          // [B, spatial..., channels]
            conditioning: MLXArray,     // from TextEncoder outlet
            conditioningMask: MLXArray, // from TextEncoder outlet
            timestep: MLXArray          // scalar or [B]
        }
outlet: MLXArray                        // noise prediction [B, spatial..., channels]
```

**Lifecycle**: Conforms to `WeightedSegment` (see R2.0). Pipeline loads weights via `WeightLoader` using the backbone's `keyMapping` and optional `tensorTransform` (e.g., PixArt provides a transform for Conv2d [O,I,kH,kW] → [O,kH,kW,I] transposition). Weights are delivered through `apply(weights:)`.

**Shape contract**: `var expectedConditioningDim: Int { get }` — must match the connected TextEncoder's `outputEmbeddingDim`. `var outputLatentChannels: Int { get }` — must match the connected Decoder's `expectedInputChannels`. Validated at pipeline assembly time.

The backbone protocol is intentionally minimal. It takes conditioned latents and a timestep, returns a noise prediction. All the iteration logic lives in the Scheduler; all the conditioning logic lives in the TextEncoder. The backbone is a pure function of its inputs.

### R2.4 Decoder

Converts the backbone's output space into a decoded representation (pixels, audio samples, video frames).

```
inlet:  MLXArray           // latents from final denoising step
outlet: DecodedOutput {
            data: MLXArray,            // [B, H, W, C] for images, [B, samples] for audio
            metadata: DecoderMetadata  // modality-specific metadata for the Renderer
        }
```

```swift
public protocol DecoderMetadata {
    var scalingFactor: Float { get }    // VAE latent scaling (e.g., 0.13025 for SDXL)
}

public struct ImageDecoderMetadata: DecoderMetadata {
    let scalingFactor: Float
}

public struct AudioDecoderMetadata: DecoderMetadata {
    let scalingFactor: Float
    let sampleRate: Int                 // e.g., 44100
}
```

Protocol with one universal property (`scalingFactor`), concrete structs per modality. Video gets its own conformance when needed.

**Lifecycle**: Conforms to `WeightedSegment` (see R2.0). Pipeline loads weights via `WeightLoader` → calls `apply(weights:)`.
**Scaling**: `var scalingFactor: Float { get }` — the Decoder applies `latents * (1.0 / scalingFactor)` internally in its `decode()` method. The pipeline passes raw latents from the last denoising step and does NOT touch the scaling factor. This co-locates scaling logic with the VAE implementation.

**Shape contract**: `var expectedInputChannels: Int { get }` — must match the connected Backbone's `outputLatentChannels`. Validated at pipeline assembly time.

### R2.4.1 BidirectionalDecoder (Image-to-Image Support)

VAEs are inherently bidirectional — the same module encodes and decodes. For image-to-image generation, the pipeline needs to encode reference images into the latent space before adding noise and denoising. This is a pipeline-level concern, not a backbone concern — the backbone receives latents and produces noise predictions regardless of where the latents came from.

```swift
/// A Decoder that can also encode (pixels → latents).
/// Required for image-to-image and inpainting generation modes.
public protocol BidirectionalDecoder: Decoder {
    /// Encode pixel data into the latent space.
    /// Input:  [B, H, W, 3] (normalized float pixels)
    /// Output: [B, H/f, W/f, C] (latents, where f is the spatial downsample factor)
    func encode(_ pixels: MLXArray) throws -> MLXArray
}
```

Models that don't support img2img keep their `Decoder` conformance unchanged. Models that do (e.g., FLUX) add `BidirectionalDecoder` conformance to their VAE decoder.

**Assembly-time validation**: If a recipe declares `supportsImageToImage = true` and the Decoder does not conform to `BidirectionalDecoder`, assembly fails with `PipelineError.incompatibleComponents`.

**Pipeline img2img flow** (see R3.2 orchestration flow):
```
1. biDecoder.encode(referenceImage) → imageLatents
2. S.addNoise(to: imageLatents, noise: random, at: startTimestep) → noisyLatents
3. S.configure(steps:, startTimestep:) → truncated timestep plan
4. Denoising loop runs on noisyLatents (backbone never knows it's img2img)
5. D.decode(finalLatents) → pixels
```

The backbone never sees reference images or knows the generation mode. It receives latents (which may have started as pure noise or as a noisy reference image) and produces noise predictions — same function either way.

### R2.5 Renderer

Converts decoded MLXArray data into the final output format. Pure data transformation — no model weights, no GPU. This is where MLXArray becomes CGImage, WAV Data, or video frames.

```
inlet:  DecodedOutput      // from Decoder outlet
outlet: RenderedOutput     // enum: .image(CGImage), .audio(Data, AudioFormat), .video(VideoFrames)
```

**Output types**:

```swift
public enum RenderedOutput: Sendable {
    case image(CGImage)
    case audio(AudioData)
    case video(VideoFrames)
}

public struct AudioData: Sendable {
    let data: Data              // WAV-encoded bytes
    let sampleRate: Int
}

public struct VideoFrames: Sendable {
    let frames: [CGImage]
    let frameRate: Double
}
```

Simple value types. No protocol overhead — these are leaf types that carry the final output.

**No lifecycle** — renderers are stateless, weightless transformers.

### R2.6 Summary: Pipe Compatibility Matrix

| Outlet (from) | Compatible Inlet (to) | Contract |
|---|---|---|
| TextEncoder → TextEncoderOutput | Backbone.conditioning + conditioningMask | Embeddings shape must match backbone's expected caption_channels |
| Scheduler → timestep | Backbone.timestep | Scalar timestep value |
| Backbone → MLXArray | Decoder inlet | Latent shape must match decoder's expected input channels |
| Decoder → DecodedOutput | Renderer inlet | Data shape + metadata determine renderer behavior |

Each connection point has a **shape contract** that is validated at pipeline assembly time, not at generation time. If a T5-XXL encoder (4096-dim) is connected to a backbone expecting 768-dim conditioning, the pipeline refuses to assemble.

---

## R3. Pipeline Composition

A Pipeline is an assembled, validated chain of pipe segments. The pipeline manages the lifecycle of all its components and orchestrates the generation flow.

### R3.1 Pipeline Protocol

```swift
public protocol GenerationPipeline: Sendable {
    associatedtype Request
    associatedtype Result

    func generate(request: Request, progress: @Sendable (PipelineProgress) -> Void) async throws -> Result
    func loadModels(progress: @Sendable (Double, String) -> Void) async throws
    func unloadModels() async
    var memoryRequirement: MemoryRequirement { get }
    var isLoaded: Bool { get }
}

public struct MemoryRequirement {
    /// Total memory if all components loaded simultaneously (Mac strategy)
    let peakMemoryBytes: UInt64

    /// Maximum memory needed for any single loading phase (iPad two-phase strategy)
    /// e.g., max(encoder phase, backbone + decoder phase)
    let phasedMemoryBytes: UInt64
}
```

MemoryManager compares `peakMemoryBytes` to available device memory. If insufficient, it falls back to phased loading using `phasedMemoryBytes`. Consumers (e.g., SwiftVinetas engines) use this for memory validation UI.

### R3.2 DiffusionPipeline

The standard pipeline for all diffusion-based generation (images, video, non-speech audio). Composed from five pipe segments:

```swift
DiffusionPipeline<
    E: TextEncoder,
    S: Scheduler,
    B: Backbone,
    D: Decoder,
    R: Renderer
>
```

**Concrete request/result types** for DiffusionPipeline:

```swift
public struct DiffusionGenerationRequest {
    let prompt: String
    let negativePrompt: String?       // nil if model doesn't use CFG negative conditioning
    let width: Int
    let height: Int
    let steps: Int
    let guidanceScale: Float
    let seed: UInt32?                 // nil → random
    let loRA: LoRAConfig?
    let referenceImages: [CGImage]?  // for image-to-image (pipeline encodes via BidirectionalDecoder)
    let strength: Float?              // denoising strength for img2img (0.0 = identity, 1.0 = full denoise)
}

public struct DiffusionGenerationResult {
    let output: RenderedOutput        // .image(CGImage), .audio(Data), etc.
    let seed: UInt32                  // actual seed used (important when request seed is nil)
    let steps: Int
    let guidanceScale: Float
    let duration: TimeInterval
}
```

**Design notes**:
- `negativePrompt` is optional — not all models use CFG with negative conditioning.
- `seed` in the result is the actual seed used, which matters when the request seed is nil (random).
- `modelID` is intentionally omitted — the pipeline doesn't know about consumer-level model descriptors. Consumers (e.g., SwiftVinetas engines) add that when translating back to their own result types.
- Image-to-image is a pipeline-level concern, not a backbone concern. If `referenceImages` is present, the pipeline encodes them via `BidirectionalDecoder.encode()`, adds noise via `Scheduler.addNoise()`, and starts the denoising loop from the noisy reference latents instead of pure noise. The backbone never knows it's doing img2img — it receives latents and produces noise predictions either way. Recipes that support img2img must use a Decoder conforming to `BidirectionalDecoder` (see R2.4.1).

**Classifier-Free Guidance (CFG)**:
Each `PipelineRecipe` declares an `UnconditionalEmbeddingStrategy`:

```swift
public enum UnconditionalEmbeddingStrategy: Sendable {
    case emptyPrompt              // encode "" through same encoder (PixArt, SD, SDXL)
    case zeroVector(shape: [Int]) // all-zero embedding of given shape
    case none                     // model uses guidance embedding, not CFG (FLUX)
}
```

- If `.emptyPrompt`: pipeline encodes `""` through the same TextEncoder for the unconditional pass
- If `.zeroVector`: pipeline constructs zeros of the declared shape
- If `.none`: skip CFG entirely — guidance scale is informational only (e.g., FLUX embeds it into the model via the backbone's conditioning)

**Orchestration flow**:
1. `E.encode(input)` → embeddings
2. **Initial latents**:
   - **Text-to-image**: Sample pure noise latents
   - **Image-to-image**: `(D as BidirectionalDecoder).encode(referenceImage)` → `S.addNoise(to: imageLatents, noise: random, at: startTimestep)` → noisy reference latents
3. `S.configure(steps:, startTimestep:)` → timestep plan (truncated for img2img based on `strength`)
4. For each timestep:
   a. Concatenate latents for CFG (if guidance > 1.0)
   b. `B.forward(latents, embeddings, timestep)` → noise prediction
   c. Apply CFG: `uncond + scale * (cond - uncond)`
   d. `S.step(prediction, timestep, latents)` → updated latents
   e. Report progress
5. `D.decode(final_latents)` → decoded output
6. `R.render(decoded)` → final output

The backbone sees the same interface in both modes — it is never aware of the generation mode.

### R3.3 Pipeline Recipes

A recipe declares which pipe segments to connect. Model plugins provide recipes:

```swift
public protocol PipelineRecipe {
    associatedtype Encoder: TextEncoder
    associatedtype Sched: Scheduler
    associatedtype Back: Backbone
    associatedtype Dec: Decoder
    associatedtype Rend: Renderer

    var encoderConfig: Encoder.Configuration { get }
    var schedulerConfig: Sched.Configuration { get }
    var backboneConfig: Back.Configuration { get }
    var decoderConfig: Dec.Configuration { get }
    var rendererConfig: Rend.Configuration { get }

    /// Whether this recipe supports image-to-image generation.
    /// If true, the Decoder must conform to BidirectionalDecoder.
    var supportsImageToImage: Bool { get }

    /// Validate that all pipe segments are compatible before assembly.
    func validate() throws
}
```

**Assembly-time validation** — `validate()` performs five checks:
1. **Completeness**: All required components are provided (non-nil)
2. **Encoder→Backbone**: `encoder.outputEmbeddingDim == backbone.expectedConditioningDim`
3. **Backbone→Decoder**: `backbone.outputLatentChannels == decoder.expectedInputChannels`
4. **Decoder→Renderer**: Decoder output modality is compatible with renderer input (image decoder → ImageRenderer, audio decoder → AudioRenderer)
5. **Image-to-image**: If `supportsImageToImage`, Decoder conforms to `BidirectionalDecoder`

Failure throws `PipelineError.incompatibleComponents(inlet:outlet:reason:)` with clear diagnostic messages. Additionally, `loadModels()` verifies all Acervo component IDs are ready before loading weights.

Each recipe also declares its `unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy` for CFG handling.

**Example — PixArt recipe**:
```
Encoder:   T5XXLEncoder          (from SwiftTubería catalog)
Scheduler: DPMSolverScheduler    (from SwiftTubería catalog)
Backbone:  PixArtDiT             (from pixart-swift-mlx — the ONLY new code)
Decoder:   SDXLVAEDecoder        (from SwiftTubería catalog)
Renderer:  ImageRenderer         (from SwiftTubería catalog)
```

**Example — FLUX.2 Klein recipe**:
```
Encoder:   Qwen3TextEncoder      (from flux-2-swift-mlx)
Scheduler: FlowMatchEulerScheduler (from SwiftTubería catalog)
Backbone:  FluxDiT               (from flux-2-swift-mlx — the unique architecture)
Decoder:   FluxVAEDecoder        (from flux-2-swift-mlx)
Renderer:  ImageRenderer         (from SwiftTubería catalog)
```

### R3.4 Two-Phase Loading

Many pipelines need to load components in phases to fit within memory budgets. The pipeline manages this transparently:

**Phase 1 — Conditioning**: Load TextEncoder → encode prompt → unload TextEncoder
**Phase 2 — Generation**: Load Backbone + Decoder → denoise → decode → unload

The pipeline's `memoryRequirement` reports both the peak memory (all components loaded) and the phased memory (max of any single phase). The consuming app can choose the strategy.

**Phase declaration**: Pipeline recipes declare their phase groupings — which components belong to which loading phase. For diffusion pipelines the standard grouping is: Phase 1 = TextEncoder, Phase 2 = Backbone + Decoder + Renderer. The pipeline executes these phases in order, calling `MemoryManager.clearGPUCache()` between phases.

---

## R4. Shared Component Catalog

Concrete implementations of pipe segments that are reused across multiple models. These live in SwiftTubería, not in model plugin packages.

### R4.1 Encoders

| Component | Embedding Dim | Used By | Size (int4) |
|---|---|---|---|
| `T5XXLEncoder` | 4096 | PixArt-Sigma, future SD3 | ~1.2 GB |
| `CLIPEncoder` | 768 | Future SDXL, SD models | ~400 MB |

Encoders that are model-specific (Qwen3 for FLUX Klein, Mistral for FLUX Dev) remain in their model plugin packages but conform to the same `TextEncoder` protocol.

### R4.2 Schedulers

| Component | Algorithm | Used By |
|---|---|---|
| `DPMSolverScheduler` | DPM-Solver++ multistep | PixArt, SD, SDXL |
| `FlowMatchEulerScheduler` | Flow matching (rectified flow) | FLUX.2 |
| `DDPMScheduler` | Denoising diffusion | Fallback / training |

### R4.3 Decoders

| Component | Latent Channels | Used By | Size |
|---|---|---|---|
| `SDXLVAEDecoder` | 4, scale 0.13025 | PixArt, SDXL, SD | ~160 MB |

Model-specific decoders (FLUX VAE) remain in their plugin packages.

### R4.4 Renderers

| Component | Input | Output | Stateless |
|---|---|---|---|
| `ImageRenderer` | [B, H, W, 3] float | CGImage | Yes |
| `AudioRenderer` | [B, samples] float | Data (WAV/M4A) | Yes |

Renderers have no model weights. They are pure data transformations.

---

## R5. Infrastructure Services

Shared services available to all pipe segments and pipelines.

### R5.1 Model Access via SwiftAcervo

**SwiftTubería does not have its own model registry.** All model discovery, download, caching, and file access is delegated to SwiftAcervo's Component Registry (see SwiftAcervo REQUIREMENTS.md).

Model plugins register their `ComponentDescriptor` entries with Acervo at import time. SwiftTubería addresses models exclusively through Acervo's abstractions:

- **Catalog queries**: `Acervo.isComponentReady(id)`, `Acervo.registeredComponents()`
- **Downloads**: `Acervo.ensureComponentsReady(recipe.allComponentIds)`
- **Weight access**: `AcervoManager.shared.withComponentAccess(id) { handle in ... }`

Pipeline code never constructs file paths, HuggingFace URLs, or hardcoded repo strings. If Acervo changes its storage layout, caching strategy, or download source — no pipeline code changes.

### R5.2 Weight Loader

Loads safetensors files into `ModuleParameters` (see R2.0) with key remapping, tensor transforms, and quantization. The single centralized loading path — no pipe segment ever parses safetensors or accesses files directly.

**Contract**:
```swift
public struct WeightLoader {
    /// Load weights for a component through Acervo, applying key remapping,
    /// optional tensor transforms, and quantization.
    public static func load(
        componentId: String,
        keyMapping: KeyMapping,
        tensorTransform: TensorTransform? = nil,
        quantization: QuantizationConfig = .asStored
    ) async throws -> ModuleParameters
}
```

**Loading pipeline** (internal to WeightLoader):
```
1. withComponentAccess(componentId) → ComponentHandle
2. handle.urls(matching: ".safetensors") → file URLs (handles sharded weights)
3. For each safetensors file, for each key:
   a. keyMapping(originalKey) → remappedKey (nil = skip)
   b. tensorTransform?(remappedKey, tensor) → transformed tensor (nil = identity)
   c. Apply quantization per QuantizationConfig
4. Collect all key-tensor pairs → ModuleParameters
```

The WeightLoader obtains file access through `AcervoManager.shared.withComponentAccess()`. It never receives or stores file paths directly. URLs are valid only within the `withComponentAccess` closure scope.

**Capabilities**:
- Safetensors parsing (single file and sharded)
- Key remapping via `KeyMapping` closure (provided by `WeightedSegment.keyMapping`)
- Per-tensor transforms via `TensorTransform` (e.g., Conv2d weight transposition, provided by `WeightedSegment.tensorTransform`)
- Dtype conversion (float32 → float16/bfloat16)
- Post-load quantization (int4, int8 with configurable group size) via `QuantizationConfig`
- Progressive loading (stream keys to reduce peak memory)

Model plugins provide their key mapping and optional tensor transform via `WeightedSegment` conformance. The WeightLoader does all the mechanical work. Pipe segments receive clean `ModuleParameters` through `apply(weights:)` and never touch file I/O.

### R5.3 Memory Manager

Coordinates memory across all loaded pipe segments. Global singleton actor.

```swift
public actor MemoryManager {
    public static let shared: MemoryManager

    // --- Device queries ---
    public var availableMemory: UInt64 { get }
    public var totalMemory: UInt64 { get }
    public var deviceCapability: DeviceCapability { get }

    // --- Budget checks ---
    /// Soft check — returns false if budget is tight, does NOT throw.
    public func softCheck(requiredBytes: UInt64) -> Bool

    /// Hard gate — throws PipelineError.insufficientMemory if not enough.
    public func hardValidate(requiredBytes: UInt64) throws

    // --- Component tracking ---
    public func registerLoaded(component: String, bytes: UInt64)
    public func unregisterLoaded(component: String)
    public var loadedComponentsMemory: UInt64 { get }

    // --- GPU cache ---
    public func clearGPUCache()
}
```

**Available memory** uses Mach VM statistics including reclaimable pages (free + inactive + purgeable + speculative), giving a realistic picture of usable memory rather than just "free" pages.

**Cross-pipeline coordination**: MemoryManager tracks all loaded components across all pipelines (image, TTS, etc.). It reports total loaded memory but does NOT auto-unload — the caller/app decides priority. If budget is tight, MemoryManager returns `false`/throws and the caller decides what to evict.

**Headroom multiplier**: Per-consumer, not in MemoryManager. Each consumer applies its own headroom (e.g., VoxAlta 1.5× for KV caches, image pipelines 1.2×) and passes the multiplied value to `softCheck`/`hardValidate`. MemoryManager provides raw memory and loaded-component tracking.

**Two-phase coordination**: Pipeline initiates, MemoryManager coordinates. Pipeline queries `softCheck(peakMemoryBytes)`. If insufficient, pipeline falls back to phased loading, calling `clearGPUCache()` between phases. MemoryManager tracks what's loaded via `registerLoaded`/`unregisterLoaded` but does not decide the phase strategy.

**Device Capability Detection**:

```swift
public struct DeviceCapability: Sendable {
    public let chipGeneration: AppleSiliconGeneration
    public let totalMemoryGB: Int
    public let platform: Platform
    public let hasNeuralAccelerators: Bool

    public enum AppleSiliconGeneration: String, Sendable, CaseIterable {
        case m1, m1Pro, m1Max, m1Ultra
        case m2, m2Pro, m2Max, m2Ultra
        case m3, m3Pro, m3Max, m3Ultra
        case m4, m4Pro, m4Max, m4Ultra
        case m5, m5Pro, m5Max, m5Ultra
        case unknown
    }

    public enum Platform: String, Sendable {
        case macOS, iPadOS
    }

    public static let current: DeviceCapability
}
```

Detection uses `sysctlbyname("machdep.cpu.brand_string")`. Neural Accelerator detection contributed from SwiftVoxAlta's M5 logic. Cached at first access.

### R5.4 Progress Reporter

Unified progress reporting for all pipeline operations.

```swift
public enum PipelineProgress: Sendable {
    case downloading(component: String, fraction: Double)
    case loading(component: String, fraction: Double)
    case encoding(fraction: Double)
    case generating(step: Int, totalSteps: Int, elapsed: TimeInterval)
    case decoding
    case rendering
    case complete(duration: TimeInterval)
}
```

---

## R6. Model Plugin Contract

What a model plugin package (e.g., pixart-swift-mlx, flux-2-swift-mlx) must provide to participate in the pipeline system.

### R6.1 Required

1. **Backbone implementation** — A struct/class conforming to the `Backbone` protocol. This is the unique neural network architecture. This is the only substantial new code per model.

2. **Model configuration** — A struct declaring the architecture parameters (hidden size, head count, depth, etc.). Weight key mapping and optional tensor transforms are provided through the `WeightedSegment` protocol conformance (`keyMapping`, `tensorTransform`).

3. **Pipeline recipe** — Declares which encoder, scheduler, decoder, and renderer to connect. References catalog components where available, provides custom components where needed.

4. **Acervo component descriptors** — Declares `ComponentDescriptor` entries for registration into SwiftAcervo's Component Registry (HuggingFace repos, file sizes, checksums).

### R6.2 Optional

5. **Custom TextEncoder** — Only if the model uses an encoder not in the catalog (e.g., FLUX's Qwen3/Mistral encoders).

6. **Custom Decoder** — Only if the model uses a decoder not in the catalog (e.g., FLUX's specific VAE).

7. **LoRA support** — Model-specific LoRA target layers. The infrastructure for loading/applying LoRA weights is in SwiftTubería; the plugin declares which layers accept adapters.

8. **Weight conversion scripts** — Python scripts to convert upstream PyTorch checkpoints to MLX safetensors format.

### R6.3 What Plugins Do NOT Provide

- Model downloading logic (SwiftAcervo handles this)
- Weight loading mechanics (WeightLoader handles this)
- Memory management (MemoryManager handles this)
- Noise scheduling (Scheduler catalog handles this)
- Image/audio rendering (Renderer catalog handles this)
- Progress reporting (PipelineProgress handles this)
- Quantization logic (WeightLoader handles this)

---

## R7. LoRA System

LoRA support is split between the pipeline and model plugins.

**SwiftTubería provides**:
- LoRA weight loading from safetensors
- LoRA application mechanics (merge adapter weights into base weights)
- LoRA scaling (0.0–1.0)
- LoRA unloading (restore base weights)

**Model plugins provide**:
- Declaration of which layers accept LoRA adapters (target layer paths)
- Any model-specific LoRA key mapping

This separation means a new model gets LoRA support by declaring its target layers — no new LoRA loading code.

**Constraint**: Single active LoRA per generation (v1). Multiple LoRAs require sequential load/unload. This matches the verified limitation from the FLUX implementation. The infrastructure can be extended to support `[LoRAConfig]` with per-adapter scaling in a future version.

---

## R8. Error Model

Errors are scoped to the pipe segment that produces them, with a pipeline-level wrapper.

```swift
public enum PipelineError: Error {
    // Assembly errors (caught before generation)
    case incompatibleComponents(inlet: String, outlet: String, reason: String)
    case missingComponent(role: String)

    // Infrastructure errors
    case modelNotDownloaded(component: String)
    case insufficientMemory(required: UInt64, available: UInt64, component: String)
    case weightLoadingFailed(component: String, reason: String)
    case downloadFailed(component: String, reason: String)

    // Generation errors
    case encodingFailed(reason: String)
    case generationFailed(step: Int, reason: String)
    case decodingFailed(reason: String)
    case renderingFailed(reason: String)

    // Cancellation
    case cancelled
}
```

---

## R9. Concurrency Model

- **Pipelines** are actors — one generation at a time per pipeline instance.
- **Pipe segments** (encoders, backbones, decoders) are `Sendable` but assume single-threaded access during forward passes. The pipeline actor serializes access.
- **Renderers** are stateless and freely concurrent.
- **SwiftAcervo's AcervoManager** and **Memory Manager** are actors — safe for concurrent queries from multiple pipelines.
- Multiple pipeline instances can exist simultaneously (e.g., one for image generation, one for audio), each with their own loaded components.

---

## R10. Dependencies

```swift
.package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.2"),
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
.package(url: "<SwiftAcervo>", from: "2.0.0"),
```

SwiftTubería depends on mlx-swift for compute, swift-transformers for tokenization, and SwiftAcervo for all model management. It does NOT depend on any model plugin package — the dependency arrow points inward.

```
pixart-swift-mlx ──▶ SwiftTubería ──▶ SwiftAcervo
flux-2-swift-mlx ──▶ SwiftTubería ──▶ SwiftAcervo
SwiftVinetas ──────▶ SwiftTubería ──▶ SwiftAcervo
SwiftVoxAlta ──────▶ SwiftTubería ──▶ SwiftAcervo

Model plugins also depend on SwiftAcervo directly (for component registration).
```

---

## R11. Package Structure

```swift
products: [
    // Core pipeline system — protocols, composition, infrastructure
    .library(name: "Tubería", targets: ["Tubería"]),

    // Shared component catalog — reusable pipe segments
    .library(name: "TuberíaCatalog", targets: ["TuberíaCatalog"]),
]
```

- **`Tubería`** — Protocols, pipeline builder, infrastructure (weight loader, memory manager, progress). Model access via SwiftAcervo. No model-specific code.
- **`TuberíaCatalog`** — Concrete shared components (T5XXLEncoder, SDXLVAEDecoder, DPMSolverScheduler, FlowMatchEulerScheduler, ImageRenderer, AudioRenderer). Depends on Tubería.

Model plugins depend on `Tubería` (for protocols) and optionally on `TuberíaCatalog` (for shared components they want to reuse).

---

## R12. Testing Strategy

### R12.1 Component Tests (per pipe segment)

Each shared component is tested in isolation:
- **T5XXLEncoder**: known prompt → expected embedding shape and values
- **SDXLVAEDecoder**: known latent tensor → expected pixel output (PSNR vs reference)
- **DPMSolverScheduler**: synthetic noise predictions → expected denoising trajectory
- **ImageRenderer**: known pixel array → valid CGImage with correct dimensions
- **AudioRenderer**: known sample array → valid WAV with correct format

### R12.2 Contract Tests (pipe compatibility)

Validate that outlet shapes match inlet expectations:
- T5XXLEncoder outlet dim (4096) matches PixArtDiT conditioning inlet (4096)
- SDXLVAEDecoder inlet channels (4) matches PixArt backbone output channels (4)
- FluxVAEDecoder inlet channels (16) matches FluxDiT output channels (16)
- Pipeline assembly with mismatched components fails with `PipelineError.incompatibleComponents`
- Recipe with `supportsImageToImage = true` and non-`BidirectionalDecoder` decoder fails at assembly
- `WeightedSegment.apply(weights:)` with missing keys throws clear error
- `WeightedSegment.apply(weights:)` with correct keys succeeds and sets `isLoaded = true`
- `WeightLoader.load()` applies `keyMapping`, `tensorTransform`, and `QuantizationConfig` correctly (synthetic safetensors with known key names and tensor shapes)

### R12.3 Integration Tests

Full pipeline smoke tests per model (provided by model plugin test suites):
- PixArt recipe: prompt → CGImage (correct dimensions, non-zero pixels)
- FLUX.2 recipe: prompt → CGImage (correct dimensions, non-zero pixels)

**Seed reproducibility thresholds**:
- Same device, same seed → PSNR > 40 dB between runs ("visually identical")
- Cross-platform (macOS vs iPadOS, different M-series) → PSNR > 30 dB (MLX float-point order may differ across GPU architectures)
- Byte-for-byte reproduction is NOT guaranteed and NOT required

**Weight conversion thresholds**:
- Converted weights (PyTorch → MLX safetensors) must produce output within PSNR > 30 dB of PyTorch reference
- Per-layer validation: investigate if any single layer drops below 25 dB (even if end-to-end passes)

### R12.4 Infrastructure Tests

- Acervo integration: component access via handles, download orchestration (tested in SwiftAcervo; Pipeline tests verify the integration seam)
- Weight Loader: safetensors parsing, key remapping via `KeyMapping`, tensor transforms via `TensorTransform`, quantization via `QuantizationConfig`, delivery via `ModuleParameters` → `apply(weights:)`
- Memory Manager: device detection, budget enforcement, phase coordination

### R12.5 Coverage and CI Stability Requirements

- All new code must achieve **≥90% line coverage** in unit tests. Coverage is measured per-target (`Tubería` and `TuberíaCatalog` separately) and enforced in CI.
- **No timed tests**: Tests must not use `sleep()`, `Task.sleep()`, `Thread.sleep()`, fixed-duration `XCTestExpectation` timeouts, or any wall-clock assertions. All asynchronous behavior must be validated via deterministic synchronization (`async`/`await`, `AsyncStream`, fulfilled expectations with immediate triggers).
- **No environment-dependent tests**: Protocol conformance tests, pipeline assembly/validation tests, scheduler math tests, and renderer data-transformation tests must use synthetic inputs and mock components — no real model weights, network access, or GPU required. Tests requiring downloaded models and GPU compute (e.g., T5XXLEncoder encoding, SDXLVAEDecoder PSNR checks) are integration tests and must be clearly separated (separate test target or `#if INTEGRATION_TESTS` gate).
- **Flaky tests are test failures**: A test that passes intermittently is treated as a failing test until fixed. CI must not use retry-on-failure to mask flakiness.

---

## R13. What This Replaces

| Previously In | Now In SwiftTubería | Stays In Original |
|---|---|---|
| flux-2-swift-mlx | Weight loading, quantization, memory management, image rendering, scheduler | FLUX DiT backbone, Qwen3/Mistral encoders, FLUX VAE, LoRA target layers |
| pixart-swift-mlx | Weight loading, quantization, memory management, image rendering, SDXL VAE, DPM-Solver, T5 encoder | PixArt DiT backbone, weight key mapping |
| SwiftVoxAlta | Model management, memory management, device detection | VoiceProvider, clone prompts, .vox handling, TTS pipeline specifics |
| SwiftVinetas | Nothing moves out — Vinetas consumes Pipeline instead of raw model libs | Engine abstraction, prompt composition, style management, PromptFile |

---

## R14. Implementation Order

1. **Tubería target** — Protocols, infrastructure (weight loader, memory manager, Acervo integration)
2. **DiffusionPipeline** — Generic diffusion orchestrator
3. **ImageRenderer** — Stateless MLXArray → CGImage (simplest renderer, immediate payoff)
4. **SDXLVAEDecoder** — First shared decoder (used by PixArt, validates decoder protocol)
5. **DPMSolverScheduler** — First shared scheduler (used by PixArt)
6. **T5XXLEncoder** — First shared encoder (used by PixArt)
7. **PixArt integration** — First model plugin, proves the system end-to-end
8. **FlowMatchEulerScheduler** — Enables FLUX.2 migration
9. **FLUX.2 migration** — Retrofit flux-2-swift-mlx as a plugin
10. **AudioRenderer** — Enables future audio diffusion models
11. **SwiftVinetas integration** — Simplify engine layer to delegate to Pipeline
12. **SwiftVoxAlta integration** — Adopt infrastructure services

---

## R15. Success Criteria

The architecture is working when:

1. **Adding a new diffusion model** requires writing only: backbone (~300 lines), config (~30 lines), key mapping (~50 lines), recipe (~20 lines). Everything else is catalog components and infrastructure.

2. **Shared components are tested once** and validated for every model that uses them. Fixing a bug in SDXLVAEDecoder fixes it for PixArt AND any future model using that VAE.

3. **Pipe segments are independently testable** with synthetic inputs. No full pipeline required to validate a single component.

4. **Pipeline assembly validates compatibility** at construction time, not at generation time. Mismatched components produce compile-time or immediate runtime errors, not silent corruption.

5. **Memory management is centralized**. One MemoryManager coordinates all loaded components. Two-phase loading works identically across all models without per-model memory code.
