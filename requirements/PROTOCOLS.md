# SwiftTubería — Pipe Segment Protocols

**Parent**: [`REQUIREMENTS.md`](../REQUIREMENTS.md) — SwiftTubería Overview
**Scope**: Typed pipe segment protocols that define the inlet/outlet contracts for every component in a generation pipeline. This document contains both the design rationale and the canonical Swift definitions.

---

## Shared Lifecycle: WeightedSegment

Pipe segments that carry model weights (TextEncoder, Backbone, Decoder) share a common lifecycle. The pipeline loads weights centrally via `WeightLoader` and delivers them to segments through this protocol. Scheduler and Renderer do NOT conform — they have no weights.

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

**Recommended conformance pattern**: `final class` with `@unchecked Sendable`. Pipe segments carry mutable state (loaded weights, isLoaded flag) but the `DiffusionPipeline` actor serializes all access — segments are never called concurrently. Using `final class` avoids `mutating` complexity on value types and `@unchecked Sendable` reflects the actor-serialized safety guarantee.

```swift
/// Example minimal conformance:
public final class MyBackbone: Backbone, @unchecked Sendable {
    public typealias Configuration = MyBackboneConfig
    private var modules: [String: MLXArray] = [:]
    public private(set) var isLoaded = false
    // ... protocol requirements ...
}
```

---

## TextEncoder

Converts raw text into dense embeddings that condition the generation.

```
inlet:  TextEncoderInput  { text: String, maxLength: Int }
outlet: TextEncoderOutput { embeddings: MLXArray [B, seq, dim], mask: MLXArray [B, seq] }
```

**Lifecycle**: Conforms to `WeightedSegment`. The pipeline calls `init(configuration:)` to instantiate the encoder, then loads weights via `WeightLoader` → calls `apply(weights:)`. Non-weight resources (tokenizers) are loaded by the encoder during `init(configuration:)` — the Configuration includes the Acervo component ID for the tokenizer, and the encoder loads it via `withComponentAccess` during initialization.

**Shape contract**: `var outputEmbeddingDim: Int { get }` — e.g., 4096 for T5-XXL, 768 for CLIP. Validated at pipeline assembly against Backbone's `expectedConditioningDim`.

Different models use different text encoders (T5-XXL for PixArt, Qwen3 for FLUX Klein, Mistral for FLUX Dev, CLIP for future SD models). The encoder protocol is the same; the implementation varies.

---

## Scheduler

Drives the iterative denoising loop for diffusion models. Stateful within a single generation, stateless between generations.

```
inlet:  steps: Int, startTimestep: Int?     (startTimestep is nil for text-to-image, set for img2img)
outlet: SchedulerPlan { timesteps: [Int], sigmas: [Float] }

per-step:
  inlet:  output: MLXArray, timestep: Int, sample: MLXArray
  outlet: MLXArray             (denoised sample)
```

Configuration parameters like `guidanceScale`, `betaSchedule`, and `predictionType` are provided via the `Scheduler.Configuration` associated type at initialization, not passed per-call to `configure()`.

`betaSchedule` is optional — DDPM-family schedulers (DPM-Solver++, DDPM) require it; flow-matching schedulers (FlowMatchEuler) ignore it and use sigma schedules instead.

**Methods**:
- `configure(steps:startTimestep:) -> SchedulerPlan` — compute timestep schedule. `startTimestep` is optional (default `nil`) — when provided (for img2img), the scheduler truncates the timestep plan to begin at that step, producing a shorter denoising trajectory. The `strength` parameter in `DiffusionGenerationRequest` maps to `startTimestep` via `Int(Float(totalSteps) * (1.0 - strength))`.
- `step(output:timestep:sample:) -> MLXArray` — single denoising step
- `addNoise(to:noise:at:) -> MLXArray` — for img2img initialization
- `reset()` — clear state between generations

**Timestep shape**: Schedulers produce scalar timesteps. The DiffusionPipeline normalizes to scalar before calling `backbone.forward()`. Backbones MUST accept both scalar and `[B]` shapes defensively.

---

## Backbone

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

**Lifecycle**: Conforms to `WeightedSegment`. Pipeline loads weights via `WeightLoader` using the backbone's `keyMapping` and optional `tensorTransform` (e.g., PixArt provides a transform for Conv2d [O,I,kH,kW] → [O,kH,kW,I] transposition). Weights are delivered through `apply(weights:)`.

**Shape contract**: `var expectedConditioningDim: Int { get }` — must match the connected TextEncoder's `outputEmbeddingDim`. `var outputLatentChannels: Int { get }` — must match the connected Decoder's `expectedInputChannels`. `var expectedMaxSequenceLength: Int { get }` — must match the connected TextEncoder's configuration `maxSequenceLength`. All three validated at pipeline assembly time.

The backbone protocol is intentionally minimal. It takes conditioned latents and a timestep, returns a noise prediction. All the iteration logic lives in the Scheduler; all the conditioning logic lives in the TextEncoder. The backbone is a pure function of its inputs.

---

## Decoder

Converts the backbone's output space into a decoded representation (pixels, audio samples, video frames).

```
inlet:  MLXArray           // latents from final denoising step
outlet: DecodedOutput {
            data: MLXArray,            // [B, H, W, C] for images, [B, samples] for audio
            metadata: DecoderMetadata  // modality-specific metadata for the Renderer
        }
```

Protocol with one universal property (`scalingFactor`), concrete structs per modality. Video gets its own conformance when needed.

**Lifecycle**: Conforms to `WeightedSegment`. Pipeline loads weights via `WeightLoader` → calls `apply(weights:)`.

**Scaling**: `var scalingFactor: Float { get }` — the Decoder applies `latents * (1.0 / scalingFactor)` internally in its `decode()` method. The pipeline passes raw latents from the last denoising step and does NOT touch the scaling factor. This co-locates scaling logic with the VAE implementation.

**Shape contract**: `var expectedInputChannels: Int { get }` — must match the connected Backbone's `outputLatentChannels`. Validated at pipeline assembly time.

### BidirectionalDecoder (Image-to-Image Support)

VAEs are inherently bidirectional — the same module encodes and decodes. For image-to-image generation, the pipeline needs to encode reference images into the latent space before adding noise and denoising. This is a pipeline-level concern, not a backbone concern.

Models that don't support img2img keep their `Decoder` conformance unchanged. Models that do (e.g., FLUX) add `BidirectionalDecoder` conformance to their VAE decoder.

**Assembly-time validation**: If a recipe declares `supportsImageToImage = true` and the Decoder does not conform to `BidirectionalDecoder`, assembly fails with `PipelineError.incompatibleComponents`.

**Pipeline img2img flow** (see [PIPELINE.md](PIPELINE.md) orchestration flow):
```
1. biDecoder.encode(referenceImage) → imageLatents
2. S.addNoise(to: imageLatents, noise: random, at: startTimestep) → noisyLatents
3. S.configure(steps:, startTimestep:) → truncated timestep plan
4. Denoising loop runs on noisyLatents (backbone never knows it's img2img)
5. D.decode(finalLatents) → pixels
```

The backbone never sees reference images or knows the generation mode.

---

## Renderer

Converts decoded MLXArray data into the final output format. Pure data transformation — no model weights, no GPU. This is where MLXArray becomes CGImage, WAV Data, or video frames.

```
inlet:  DecodedOutput      // from Decoder outlet
outlet: RenderedOutput     // enum: .image(CGImage), .audio(Data, AudioFormat), .video(VideoFrames)
```

Simple value types. No protocol overhead — these are leaf types that carry the final output.

**No lifecycle** — renderers are stateless, weightless transformers.

---

## Pipe Compatibility Matrix

| Outlet (from) | Compatible Inlet (to) | Contract |
|---|---|---|
| TextEncoder → TextEncoderOutput | Backbone.conditioning + conditioningMask | Embeddings shape must match backbone's expected caption_channels |
| Scheduler → timestep | Backbone.timestep | Scalar timestep value |
| Backbone → MLXArray | Decoder inlet | Latent shape must match decoder's expected input channels |
| Decoder → DecodedOutput | Renderer inlet | Data shape + metadata determine renderer behavior |

Each connection point has a **shape contract** that is validated at pipeline assembly time, not at generation time. If a T5-XXL encoder (4096-dim) is connected to a backbone expecting 768-dim conditioning, the pipeline refuses to assemble.

---

## Canonical Swift Definitions

These are the **authoritative** protocol definitions. If any prose or pseudocode above differs from the code below, **this code governs**.

All protocols, structs, and enums below are `public` and live in the `Tubería` target.

```swift
import MLX

// MARK: - Shared Lifecycle

/// Remapped, quantized parameter tensors ready for module assignment.
public struct ModuleParameters: Sendable {
    public let parameters: [String: MLXArray]
}

/// Key remapping function: safetensors key → module key. Return nil to skip a key.
public typealias KeyMapping = @Sendable (String) -> String?

/// Optional per-tensor transform applied after key remapping, before quantization.
public typealias TensorTransform = @Sendable (String, MLXArray) -> MLXArray

/// Quantization strategy for weight loading.
public enum QuantizationConfig: Sendable {
    case asStored
    case float16
    case bfloat16
    case int4(groupSize: Int = 64)
    case int8(groupSize: Int = 64)
}

/// Lifecycle for pipe segments that carry model weights.
/// Conformers MUST be `final class` with `@unchecked Sendable`.
/// The DiffusionPipeline actor serializes all access.
public protocol WeightedSegment: Sendable {
    func apply(weights: ModuleParameters) throws
    func unload()
    var estimatedMemoryBytes: Int { get }
    var isLoaded: Bool { get }
    var keyMapping: KeyMapping { get }
    var tensorTransform: TensorTransform? { get }
}

extension WeightedSegment {
    public var tensorTransform: TensorTransform? { nil }
}

// MARK: - TextEncoder

public struct TextEncoderInput: Sendable {
    public let text: String
    public let maxLength: Int

    public init(text: String, maxLength: Int) {
        self.text = text
        self.maxLength = maxLength
    }
}

public struct TextEncoderOutput: Sendable {
    /// Dense embeddings. Shape: [B, seq, dim]
    public let embeddings: MLXArray
    /// Attention mask. Shape: [B, seq]. 1 = real token, 0 = padding.
    public let mask: MLXArray

    public init(embeddings: MLXArray, mask: MLXArray) {
        self.embeddings = embeddings
        self.mask = mask
    }
}

public protocol TextEncoder: WeightedSegment {
    /// Configuration type — must include the Acervo component ID for weights
    /// and any non-weight resources (e.g., tokenizer component ID).
    associatedtype Configuration: Sendable

    /// Construct the encoder from its configuration. The pipeline calls this
    /// during `DiffusionPipeline.init(recipe:)` to instantiate the component.
    init(configuration: Configuration) throws

    /// Embedding dimension of the encoder output. Used for assembly-time validation
    /// against `Backbone.expectedConditioningDim`.
    var outputEmbeddingDim: Int { get }

    /// Maximum sequence length this encoder produces. Used for assembly-time validation
    /// against `Backbone.expectedMaxSequenceLength`.
    var maxSequenceLength: Int { get }

    /// Encode text into dense embeddings.
    func encode(_ input: TextEncoderInput) throws -> TextEncoderOutput
}

// MARK: - Scheduler

public enum BetaSchedule: Sendable {
    case linear(betaStart: Float, betaEnd: Float)
    case cosine
    case sqrt
}

public enum PredictionType: String, Sendable {
    case epsilon    // predict noise (PixArt, SD, SDXL)
    case velocity   // predict velocity / v-prediction (FLUX)
    case sample     // predict clean sample directly
}

public struct SchedulerPlan: Sendable {
    public let timesteps: [Int]
    public let sigmas: [Float]

    public init(timesteps: [Int], sigmas: [Float]) {
        self.timesteps = timesteps
        self.sigmas = sigmas
    }
}

public protocol Scheduler: Sendable {
    /// Configuration type for scheduler initialization.
    associatedtype Configuration: Sendable

    /// Construct the scheduler from its configuration. The pipeline calls this
    /// during `DiffusionPipeline.init(recipe:)` to instantiate the component.
    init(configuration: Configuration)

    /// Compute the timestep schedule for a generation run.
    /// - Parameters:
    ///   - steps: Total number of denoising steps.
    ///   - startTimestep: Optional starting timestep for img2img (truncates the plan).
    ///     `nil` = full schedule (text-to-image). Derived from `strength` via
    ///     `Int(Float(steps) * (1.0 - strength))`.
    /// - Returns: A plan containing timesteps and sigmas for the denoising loop.
    func configure(steps: Int, startTimestep: Int?) -> SchedulerPlan

    /// Perform a single denoising step.
    /// - Parameters:
    ///   - output: Model noise prediction from the backbone.
    ///   - timestep: Current timestep.
    ///   - sample: Current noisy latents.
    /// - Returns: Updated (less noisy) latents.
    func step(output: MLXArray, timestep: Int, sample: MLXArray) -> MLXArray

    /// Add noise to clean latents at a given timestep. Used for img2img initialization.
    func addNoise(to sample: MLXArray, noise: MLXArray, at timestep: Int) -> MLXArray

    /// Clear internal state between generation runs.
    func reset()
}

extension Scheduler {
    public func configure(steps: Int) -> SchedulerPlan {
        configure(steps: steps, startTimestep: nil)
    }
}

// MARK: - Backbone

public struct BackboneInput: Sendable {
    /// Noisy latents. Shape: [B, spatial..., channels]
    public let latents: MLXArray
    /// Text encoder embeddings (mapped from TextEncoderOutput.embeddings).
    /// Shape: [B, seq, dim]
    public let conditioning: MLXArray
    /// Text encoder mask (mapped from TextEncoderOutput.mask).
    /// Shape: [B, seq]
    public let conditioningMask: MLXArray
    /// Current denoising timestep. Scalar or [B].
    public let timestep: MLXArray

    public init(latents: MLXArray, conditioning: MLXArray,
                conditioningMask: MLXArray, timestep: MLXArray) {
        self.latents = latents
        self.conditioning = conditioning
        self.conditioningMask = conditioningMask
        self.timestep = timestep
    }
}

public protocol Backbone: WeightedSegment {
    /// Configuration type — must include the Acervo component ID for weights.
    associatedtype Configuration: Sendable

    /// Construct the backbone from its configuration. The pipeline calls this
    /// during `DiffusionPipeline.init(recipe:)` to instantiate the component.
    init(configuration: Configuration) throws

    /// Expected embedding dimension from the connected TextEncoder.
    /// Validated at assembly: must equal `TextEncoder.outputEmbeddingDim`.
    var expectedConditioningDim: Int { get }

    /// Number of latent channels produced by the backbone.
    /// Validated at assembly: must equal `Decoder.expectedInputChannels`.
    var outputLatentChannels: Int { get }

    /// Maximum sequence length the backbone expects from the TextEncoder.
    /// Validated at assembly: must equal the TextEncoder configuration's `maxSequenceLength`.
    /// This ensures the encoder won't produce sequences longer than the backbone can attend to.
    var expectedMaxSequenceLength: Int { get }

    /// Forward pass — noise prediction.
    /// - Parameter input: Conditioned latents and timestep.
    /// - Returns: Noise prediction. Shape: [B, spatial..., channels]
    func forward(_ input: BackboneInput) throws -> MLXArray
}

// MARK: - Decoder

public protocol DecoderMetadata: Sendable {
    var scalingFactor: Float { get }
}

public struct ImageDecoderMetadata: DecoderMetadata, Sendable {
    public let scalingFactor: Float
    public init(scalingFactor: Float) { self.scalingFactor = scalingFactor }
}

public struct AudioDecoderMetadata: DecoderMetadata, Sendable {
    public let scalingFactor: Float
    public let sampleRate: Int
    public init(scalingFactor: Float, sampleRate: Int) {
        self.scalingFactor = scalingFactor
        self.sampleRate = sampleRate
    }
}

public struct DecodedOutput: Sendable {
    /// Decoded data. Shape: [B, H, W, C] for images, [B, samples] for audio.
    public let data: MLXArray
    /// Modality-specific metadata for the Renderer.
    public let metadata: any DecoderMetadata

    public init(data: MLXArray, metadata: any DecoderMetadata) {
        self.data = data
        self.metadata = metadata
    }
}

public protocol Decoder: WeightedSegment {
    /// Configuration type — must include the Acervo component ID for weights.
    associatedtype Configuration: Sendable

    /// Construct the decoder from its configuration. The pipeline calls this
    /// during `DiffusionPipeline.init(recipe:)` to instantiate the component.
    init(configuration: Configuration) throws

    /// Expected number of latent channels from the connected Backbone.
    /// Validated at assembly: must equal `Backbone.outputLatentChannels`.
    var expectedInputChannels: Int { get }

    /// VAE latent scaling factor. Applied internally by the decoder
    /// (`latents * (1.0 / scalingFactor)`) — the pipeline does NOT touch this.
    var scalingFactor: Float { get }

    /// Decode latents into output data.
    func decode(_ latents: MLXArray) throws -> DecodedOutput
}

/// A Decoder that can also encode (pixels → latents).
/// Required for image-to-image and inpainting generation modes.
public protocol BidirectionalDecoder: Decoder {
    /// Encode pixel data into the latent space.
    /// Input:  [B, H, W, 3] (normalized float pixels)
    /// Output: [B, H/f, W/f, C] (latents)
    func encode(_ pixels: MLXArray) throws -> MLXArray
}

// MARK: - Renderer

public enum RenderedOutput: Sendable {
    case image(CGImage)
    case audio(AudioData)
    case video(VideoFrames)
}

public struct AudioData: Sendable {
    public let data: Data
    public let sampleRate: Int
    public init(data: Data, sampleRate: Int) {
        self.data = data
        self.sampleRate = sampleRate
    }
}

public struct VideoFrames: Sendable {
    public let frames: [CGImage]
    public let frameRate: Double
    public init(frames: [CGImage], frameRate: Double) {
        self.frames = frames
        self.frameRate = frameRate
    }
}

public protocol Renderer: Sendable {
    /// Configuration type (`Void` for stateless renderers like ImageRenderer).
    associatedtype Configuration: Sendable

    /// Construct the renderer from its configuration. The pipeline calls this
    /// during `DiffusionPipeline.init(recipe:)` to instantiate the component.
    init(configuration: Configuration)

    /// Render decoded output into the final format.
    func render(_ input: DecodedOutput) throws -> RenderedOutput
}
```
