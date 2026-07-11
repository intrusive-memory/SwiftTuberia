# SwiftTuberia — Pipeline Composition

**Parent**: [`REQUIREMENTS.md`](../REQUIREMENTS.md) — SwiftTuberia Overview
**Scope**: Pipeline protocol, DiffusionPipeline orchestration, recipe system, two-phase loading, LoRA support, error model, and all pipeline-level types. This document contains both the design rationale and the canonical Swift definitions.

---

## Pipeline Protocol

A Pipeline is an assembled, validated chain of pipe segments. The pipeline manages the lifecycle of all its components and orchestrates the generation flow.

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
```

MemoryManager compares `peakMemoryBytes` to available device memory. If insufficient, it falls back to phased loading using `phasedMemoryBytes`. Consumers (e.g., SwiftVinetas engines) use this for memory validation UI.

---

## DiffusionPipeline

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

### Request and Result Types

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
- Image-to-image is a pipeline-level concern, not a backbone concern. If `referenceImages` is present, the pipeline encodes them via `BidirectionalDecoder.encode()`, adds noise via `Scheduler.addNoise()`, and starts the denoising loop from the noisy reference latents instead of pure noise. The backbone never knows it's doing img2img. Recipes that support img2img must use a Decoder conforming to `BidirectionalDecoder` (see [PROTOCOLS.md](PROTOCOLS.md)).

### Classifier-Free Guidance (CFG)

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

### Orchestration Flow

**Field mapping**: The pipeline maps `TextEncoderOutput` fields to `BackboneInput` fields during orchestration:
- `TextEncoderOutput.embeddings` → `BackboneInput.conditioning`
- `TextEncoderOutput.mask` → `BackboneInput.conditioningMask`

This mapping is internal to the pipeline — neither the encoder nor the backbone needs to know the other's field names.

**Full orchestration**:
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

---

## Pipeline Recipes

A recipe declares which pipe segments to connect. Model plugins provide recipes.

**Assembly-time validation** — `validate()` performs six checks:
1. **Completeness**: All required components are provided (non-nil)
2. **Encoder→Backbone (dim)**: `encoder.outputEmbeddingDim == backbone.expectedConditioningDim`
3. **Encoder→Backbone (seq)**: `encoder.maxSequenceLength == backbone.expectedMaxSequenceLength`
4. **Backbone→Decoder**: `backbone.outputLatentChannels == decoder.expectedInputChannels`
5. **Decoder→Renderer**: Decoder output modality is compatible with renderer input (image decoder → ImageRenderer, audio decoder → AudioRenderer)
6. **Image-to-image**: If `supportsImageToImage`, Decoder conforms to `BidirectionalDecoder`

Failure throws `PipelineError.incompatibleComponents(inlet:outlet:reason:)` with clear diagnostic messages. Additionally, `loadModels()` verifies all Acervo component IDs are ready before loading weights.

Each recipe also declares its `unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy` for CFG handling.

**Example — PixArt recipe**:
```
Encoder:   T5XXLEncoder          (from SwiftTuberia catalog)
Scheduler: DPMSolverScheduler    (from SwiftTuberia catalog)
Backbone:  PixArtDiT             (from pixart-swift-mlx — the ONLY new code)
Decoder:   SDXLVAEDecoder        (from SwiftTuberia catalog)
Renderer:  ImageRenderer         (from SwiftTuberia catalog)
```

**Illustrative — FLUX.2 Klein recipe** (deferred, shown for design validation):
```
Encoder:   Qwen3TextEncoder      (from flux-2-swift-mlx)
Scheduler: FlowMatchEulerScheduler (from SwiftTuberia catalog)
Backbone:  FluxDiT               (from flux-2-swift-mlx — the unique architecture)
Decoder:   FluxVAEDecoder        (from flux-2-swift-mlx)
Renderer:  ImageRenderer         (from SwiftTuberia catalog)
```

### DiffusionPipeline Construction API

A recipe becomes a pipeline through `DiffusionPipeline.init(recipe:)`. The init:
1. Instantiates each component via `init(configuration:)` using the recipe's config values
2. Calls `recipe.validate()` on the assembled components
3. Throws `PipelineError.incompatibleComponents` if shape contracts fail

**Engine usage pattern** (e.g., in SwiftVinetas):
```swift
// In PixArtEngine (~50 lines total)
let recipe = PixArtRecipe()  // from pixart-swift-mlx
let pipeline = try DiffusionPipeline(recipe: recipe)
try await pipeline.loadModels { fraction, component in
    reportProgress(.loading(component: component, fraction: fraction))
}
let result = try await pipeline.generate(request: diffusionRequest) { progress in
    reportProgress(progress)
}
```

---

## Two-Phase Loading

Many pipelines need to load components in phases to fit within memory budgets. The pipeline manages this transparently:

**Phase 1 — Conditioning**: Load TextEncoder → encode prompt → unload TextEncoder
**Phase 2 — Generation**: Load Backbone + Decoder → denoise → decode → unload

The pipeline's `memoryRequirement` reports both the peak memory (all components loaded) and the phased memory (max of any single phase). The consuming app can choose the strategy.

**Phase declaration**: Pipeline recipes declare their phase groupings — which components belong to which loading phase. For diffusion pipelines the standard grouping is: Phase 1 = TextEncoder, Phase 2 = Backbone + Decoder + Renderer. The pipeline executes these phases in order, calling `MemoryManager.clearGPUCache()` between phases.

---

## LoRA System

LoRA adapters are identified via `LoRAConfig`. The config accepts either an Acervo `componentId` (preferred) or a `localPath` (fallback for unregistered adapters). If both are provided, `componentId` takes precedence and `localPath` is ignored. At least one must be non-nil (enforced by precondition).

LoRA support is split between the pipeline and model plugins.

**SwiftTuberia provides**:
- LoRA weight loading from safetensors
- LoRA application mechanics (merge adapter weights into base weights)
- LoRA scaling (0.0–1.0)
- LoRA unloading (restore base weights)

**Model plugins provide**:
- LoRA key mapping — the backbone's `keyMapping` already handles safetensors key → module key translation, and LoRA adapters follow the same key namespace (e.g., `blocks.0.attn.q_proj.lora_A`, `blocks.0.attn.q_proj.lora_B`)

**LoRA application strategy**: SwiftTuberia applies LoRA adapters to **all keys present in the LoRA safetensors file** that match keys in the loaded model. No explicit target layer declaration is needed — the LoRA file itself defines which layers are adapted. The backbone's `keyMapping` is reused to translate LoRA key names to module paths. This matches the standard LoRA convention and requires zero per-model LoRA code beyond the key mapping already provided for base weights.

This separation means a new model gets LoRA support by providing its key mapping — no new LoRA loading code.

**Constraint**: Single active LoRA per generation (v1). Multiple LoRAs require sequential load/unload. This matches the verified limitation from the FLUX implementation. The infrastructure can be extended to support `[LoRAConfig]` with per-adapter scaling in a future version.

---

## Error Model

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

## Progress Reporting

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

## Canonical Swift Definitions

These are the **authoritative** pipeline type definitions. If any prose above differs from the code below, **this code governs**.

All types below are `public` and live in the `Tuberia` target.

```swift
// MARK: - Pipeline Protocol

public struct MemoryRequirement: Sendable {
    /// Total memory if all components loaded simultaneously.
    public let peakMemoryBytes: UInt64
    /// Maximum memory needed for any single loading phase.
    public let phasedMemoryBytes: UInt64

    public init(peakMemoryBytes: UInt64, phasedMemoryBytes: UInt64) {
        self.peakMemoryBytes = peakMemoryBytes
        self.phasedMemoryBytes = phasedMemoryBytes
    }
}

public protocol GenerationPipeline: Sendable {
    associatedtype Request
    associatedtype Result

    func generate(request: Request, progress: @Sendable (PipelineProgress) -> Void) async throws -> Result
    func loadModels(progress: @Sendable (Double, String) -> Void) async throws
    func unloadModels() async
    var memoryRequirement: MemoryRequirement { get }
    var isLoaded: Bool { get }
}

// MARK: - Diffusion-Specific Types

public struct DiffusionGenerationRequest: Sendable {
    public let prompt: String
    public let negativePrompt: String?
    public let width: Int
    public let height: Int
    public let steps: Int
    public let guidanceScale: Float
    public let seed: UInt32?
    public let loRA: LoRAConfig?
    public let referenceImages: [CGImage]?
    public let strength: Float?

    public init(prompt: String, negativePrompt: String? = nil,
                width: Int, height: Int, steps: Int, guidanceScale: Float,
                seed: UInt32? = nil, loRA: LoRAConfig? = nil,
                referenceImages: [CGImage]? = nil, strength: Float? = nil) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.loRA = loRA
        self.referenceImages = referenceImages
        self.strength = strength
    }
}

public struct DiffusionGenerationResult: Sendable {
    public let output: RenderedOutput
    public let seed: UInt32
    public let steps: Int
    public let guidanceScale: Float
    public let duration: TimeInterval

    public init(output: RenderedOutput, seed: UInt32, steps: Int,
                guidanceScale: Float, duration: TimeInterval) {
        self.output = output
        self.seed = seed
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.duration = duration
    }
}

public struct LoRAConfig: Sendable {
    /// Acervo component ID for the LoRA adapter safetensors.
    /// Takes precedence over `localPath` if both are provided.
    public let componentId: String?
    /// Local file path — fallback for adapters not registered in Acervo.
    /// Ignored if `componentId` is non-nil.
    public let localPath: String?
    /// Adapter scale (0.0 = no effect, 1.0 = full effect).
    public let scale: Float
    /// Optional activation keyword to prepend to the prompt.
    public let activationKeyword: String?

    /// At least one of `componentId` or `localPath` must be non-nil.
    /// If both are provided, `componentId` takes precedence and `localPath` is ignored.
    public init(componentId: String? = nil, localPath: String? = nil,
                scale: Float = 1.0, activationKeyword: String? = nil) {
        precondition(componentId != nil || localPath != nil,
                     "LoRAConfig requires at least one of componentId or localPath")
        self.componentId = componentId
        self.localPath = localPath
        self.scale = scale
        self.activationKeyword = activationKeyword
    }
}

public enum UnconditionalEmbeddingStrategy: Sendable {
    case emptyPrompt
    case zeroVector(shape: [Int])
    case none
}

// MARK: - Pipeline Recipe

public enum PipelineRole: String, Sendable, CaseIterable {
    case encoder, scheduler, backbone, decoder, renderer
}

public protocol PipelineRecipe: Sendable {
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

    var supportsImageToImage: Bool { get }
    var unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy { get }
    var allComponentIds: [String] { get }

    func quantizationFor(_ role: PipelineRole) -> QuantizationConfig
    func validate() throws
}

// MARK: - DiffusionPipeline

public actor DiffusionPipeline<E: TextEncoder, S: Scheduler, B: Backbone, D: Decoder, R: Renderer>: GenerationPipeline {
    public typealias Request = DiffusionGenerationRequest
    public typealias Result = DiffusionGenerationResult

    /// Construct a pipeline from a recipe. Calls `recipe.validate()` during construction.
    /// Throws `PipelineError.incompatibleComponents` if validation fails.
    public init<Recipe: PipelineRecipe>(recipe: Recipe) throws
        where Recipe.Encoder == E, Recipe.Sched == S,
              Recipe.Back == B, Recipe.Dec == D, Recipe.Rend == R

    public func generate(request: Request,
                         progress: @Sendable (PipelineProgress) -> Void) async throws -> Result
    public func loadModels(progress: @Sendable (Double, String) -> Void) async throws
    public func unloadModels() async

    /// Memory requirements — computed from static config, safe to access without `await`.
    nonisolated public var memoryRequirement: MemoryRequirement { get }
    /// Whether all weighted segments currently have weights loaded.
    /// Computed from immutable recipe data + segment isLoaded flags.
    nonisolated public var isLoaded: Bool { get }
}

// MARK: - Progress & Errors

public enum PipelineProgress: Sendable {
    case downloading(component: String, fraction: Double)
    case loading(component: String, fraction: Double)
    case encoding(fraction: Double)
    case generating(step: Int, totalSteps: Int, elapsed: TimeInterval)
    case decoding
    case rendering
    case complete(duration: TimeInterval)
}

public enum PipelineError: Error {
    case incompatibleComponents(inlet: String, outlet: String, reason: String)
    case missingComponent(role: String)
    case modelNotDownloaded(component: String)
    case insufficientMemory(required: UInt64, available: UInt64, component: String)
    case weightLoadingFailed(component: String, reason: String)
    case downloadFailed(component: String, reason: String)
    case encodingFailed(reason: String)
    case generationFailed(step: Int, reason: String)
    case decodingFailed(reason: String)
    case renderingFailed(reason: String)
    case cancelled
}
```
