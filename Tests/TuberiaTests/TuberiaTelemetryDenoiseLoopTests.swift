import CoreGraphics
import Foundation
@preconcurrency import MLX
import MLXRandom
import Testing

@testable import Tuberia

// MARK: - §7 Row 2 — Denoise loop telemetry
//
// Tests:
//   - Run 4 deterministic denoise steps with MockBackbone/MockScheduler.
//   - Assert: 4× denoiseStepStart, 4× denoiseStepComplete, stepIndex monotone,
//     latentAfterStat.shape matches the configured latent shape.

// MARK: - Shared Mock Components (used by denoise, CFG-cast, and anomaly tests)
//
// These are `internal` so they can be imported by test files in this target
// via `@testable import`. The mocks are kept lightweight — no real MLX model
// weights, deterministic shapes, pre-loaded so generate() does not throw
// missingComponent.

final class DenoiseEncoder: TextEncoder, @unchecked Sendable {
  struct Config: Sendable {
    let dim: Int
    let seqLen: Int
  }
  typealias Configuration = Config
  var isLoaded: Bool = true  // Pre-loaded — no weight files needed
  private var _weights: ModuleParameters? = ModuleParameters(parameters: [:])
  var outputEmbeddingDim: Int { configuration.dim }
  var maxSequenceLength: Int { configuration.seqLen }
  var estimatedMemoryBytes: Int { 0 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { _weights }
  private let configuration: Config
  required init(configuration: Config) throws { self.configuration = configuration }
  func encode(_ input: TextEncoderInput) throws -> TextEncoderOutput {
    // Return float16 embeddings to exercise the CFG cast path
    TextEncoderOutput(
      embeddings: MLXArray.zeros([1, input.maxLength, configuration.dim]).asType(.float16),
      mask: MLXArray.ones([1, input.maxLength]).asType(.float16)
    )
  }
  func apply(weights: ModuleParameters) throws {
    _weights = weights
    isLoaded = true
  }
  func unload() {
    _weights = nil
    isLoaded = false
  }
}

/// Deterministic backbone: returns a zero tensor of the latent shape.
/// Subclasses / wrappers can override `forward` to inject anomalies.
class DenoiseBackbone: Backbone, @unchecked Sendable {
  struct Config: Sendable {
    let condDim: Int
    let seqLen: Int
    let latentChannels: Int
  }
  typealias Configuration = Config
  var isLoaded: Bool = true
  private var _weights: ModuleParameters? = ModuleParameters(parameters: [:])
  var expectedConditioningDim: Int { configuration.condDim }
  var outputLatentChannels: Int { configuration.latentChannels }
  var expectedMaxSequenceLength: Int { configuration.seqLen }
  var estimatedMemoryBytes: Int { 0 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { _weights }
  let configuration: Config
  required init(configuration: Config) throws { self.configuration = configuration }
  func forward(_ input: BackboneInput) throws -> MLXArray {
    // Returns float16 zeros — same shape as latents
    MLXArray.zeros(input.latents.shape).asType(.float16)
  }
  func apply(weights: ModuleParameters) throws {
    _weights = weights
    isLoaded = true
  }
  func unload() {
    _weights = nil
    isLoaded = false
  }
}

final class DenoiseDecoder: Decoder, @unchecked Sendable {
  struct Config: Sendable { let inputChannels: Int }
  typealias Configuration = Config
  var isLoaded: Bool = true
  private var _weights: ModuleParameters? = ModuleParameters(parameters: [:])
  var expectedInputChannels: Int { configuration.inputChannels }
  var scalingFactor: Float { 0.18215 }
  var estimatedMemoryBytes: Int { 0 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { _weights }
  private let configuration: Config
  required init(configuration: Config) throws { self.configuration = configuration }
  func decode(_ latents: MLXArray) throws -> DecodedOutput {
    let shape = latents.shape
    let b = shape[0]
    let h = shape.count > 1 ? shape[1] * 8 : 8
    let w = shape.count > 2 ? shape[2] * 8 : 8
    return DecodedOutput(
      data: MLXArray.zeros([b, h, w, 3]).asType(.float32),
      metadata: ImageDecoderMetadata(scalingFactor: scalingFactor)
    )
  }
  func apply(weights: ModuleParameters) throws {
    _weights = weights
    isLoaded = true
  }
  func unload() {
    _weights = nil
    isLoaded = false
  }
}

/// Deterministic scheduler: returns `steps` evenly-spaced timesteps and all-1 sigmas.
final class DenoiseScheduler: Scheduler, @unchecked Sendable {
  struct Config: Sendable { let predType: String }
  typealias Configuration = Config
  private let configuration: Config
  required init(configuration: Config) { self.configuration = configuration }
  func configure(steps: Int, startTimestep: Int?) -> SchedulerPlan {
    // Simple descending timesteps: [999, 749, 499, 249] for 4 steps, etc.
    let stride = max(1, 1000 / steps)
    let timesteps = (0..<steps).map { 999 - $0 * stride }
    let sigmas = [Float](repeating: 1.0, count: steps)
    return SchedulerPlan(timesteps: timesteps, sigmas: sigmas)
  }
  func step(output: MLXArray, timestep: Int, sample: MLXArray) throws -> MLXArray {
    // Simple Euler step: sample - output (deterministic, no randomness)
    sample - output.asType(sample.dtype)
  }
  func addNoise(to sample: MLXArray, noise: MLXArray, at timestep: Int) -> MLXArray { sample }
  func reset() {}
  var predictionType: String { configuration.predType }
}

final class DenoiseRenderer: Renderer, @unchecked Sendable {
  typealias Configuration = Void
  required init(configuration: Void) {}
  func render(_ input: DecodedOutput) throws -> RenderedOutput {
    var pixelData: [UInt8] = [0, 0, 0, 255]
    let space = CGColorSpaceCreateDeviceRGB()
    guard
      let ctx = CGContext(
        data: &pixelData, width: 1, height: 1,
        bitsPerComponent: 8, bytesPerRow: 4, space: space,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      ),
      let img = ctx.makeImage()
    else {
      throw PipelineError.renderingFailed(reason: "DenoiseRenderer: CGImage creation failed")
    }
    return .image(img)
  }
}

// MARK: - Recipe (no-CFG, guidanceScale <= 1.0)

struct DenoiseNoCFGRecipe: PipelineRecipe, Sendable {
  typealias Encoder = DenoiseEncoder
  typealias Sched = DenoiseScheduler
  typealias Back = DenoiseBackbone
  typealias Dec = DenoiseDecoder
  typealias Rend = DenoiseRenderer

  let encoderConfig = DenoiseEncoder.Config(dim: 64, seqLen: 8)
  let schedulerConfig = DenoiseScheduler.Config(predType: "epsilon")
  let backboneConfig = DenoiseBackbone.Config(condDim: 64, seqLen: 8, latentChannels: 4)
  let decoderConfig = DenoiseDecoder.Config(inputChannels: 4)
  let rendererConfig: Void = ()

  let supportsImageToImage = false
  let unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .emptyPrompt
  let allComponentIds: [String] = []
  func quantizationFor(_ role: PipelineRole) -> QuantizationConfig { .asStored }
  func validate() throws {}
}

/// CFG-enabled recipe: unconditionalEmbeddingStrategy = .emptyPrompt and guidanceScale > 1.0
struct DenoiseCFGRecipe: PipelineRecipe, Sendable {
  typealias Encoder = DenoiseEncoder
  typealias Sched = DenoiseScheduler
  typealias Back = DenoiseBackbone
  typealias Dec = DenoiseDecoder
  typealias Rend = DenoiseRenderer

  let encoderConfig = DenoiseEncoder.Config(dim: 64, seqLen: 8)
  let schedulerConfig = DenoiseScheduler.Config(predType: "epsilon")
  let backboneConfig = DenoiseBackbone.Config(condDim: 64, seqLen: 8, latentChannels: 4)
  let decoderConfig = DenoiseDecoder.Config(inputChannels: 4)
  let rendererConfig: Void = ()

  let supportsImageToImage = false
  let unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .emptyPrompt
  let allComponentIds: [String] = []
  func quantizationFor(_ role: PipelineRole) -> QuantizationConfig { .asStored }
  func validate() throws {}
}

// MARK: - Pipeline type aliases

typealias DenoisePipeline = DiffusionPipeline<
  DenoiseEncoder,
  DenoiseScheduler,
  DenoiseBackbone,
  DenoiseDecoder,
  DenoiseRenderer
>

// MARK: - Tests

@Suite("TuberiaTelemetryDenoiseLoopTests — §7 row 2", .serialized)
struct TuberiaTelemetryDenoiseLoopTests {

  // MARK: - Setup helpers

  /// Build a no-CFG pipeline with the recorder attached.
  private func makePipeline(rec: RecordingTelemetryReporter) async throws -> DenoisePipeline {
    let pipeline = try DenoisePipeline(recipe: DenoiseNoCFGRecipe(), telemetry: rec)
    // Skip loadModels — components are pre-loaded
    await pipeline.setMemoryGate { _ in /* no-op */ }
    return pipeline
  }

  // MARK: - 4-step denoise assertions

  @Test("4× denoiseStepStart and 4× denoiseStepComplete fire in 4-step run")
  func fourStepDenoiseEmitsFourStartAndFourComplete() async throws {
    MLXRandom.seed(42)
    let rec = RecordingTelemetryReporter()
    let pipeline = try await makePipeline(rec: rec)

    let request = DiffusionGenerationRequest(
      prompt: "a test prompt",
      width: 64, height: 64,
      steps: 4,
      guidanceScale: 1.0,  // no CFG
      seed: 0
    )
    _ = try await pipeline.generate(request: request, progress: { _ in })

    let events = await rec.events

    let starts = events.compactMap { e -> Int? in
      if case .denoiseStepStart(let idx, _, _, _, _, _) = e { return idx } else { return nil }
    }
    let completes = events.compactMap { e -> Int? in
      if case .denoiseStepComplete(let idx, _, _, _, _, _, _) = e { return idx } else { return nil }
    }

    #expect(starts.count == 4, "Expected 4 denoiseStepStart events, got \(starts.count)")
    #expect(completes.count == 4, "Expected 4 denoiseStepComplete events, got \(completes.count)")
  }

  @Test("stepIndex is monotonically increasing across denoiseStepStart events")
  func stepIndexIsMonotone() async throws {
    MLXRandom.seed(42)
    let rec = RecordingTelemetryReporter()
    let pipeline = try await makePipeline(rec: rec)

    let request = DiffusionGenerationRequest(
      prompt: "monotone test",
      width: 64, height: 64,
      steps: 4,
      guidanceScale: 1.0,
      seed: 1
    )
    _ = try await pipeline.generate(request: request, progress: { _ in })

    let events = await rec.events
    let indices = events.compactMap { e -> Int? in
      if case .denoiseStepStart(let idx, _, _, _, _, _) = e { return idx } else { return nil }
    }

    #expect(indices.count == 4)
    for i in 1..<indices.count {
      #expect(indices[i] > indices[i - 1], "stepIndex not monotone at position \(i)")
    }
  }

  @Test("latentAfterStat.shape matches configured latent shape")
  func latentAfterStatShapeMatchesLatentShape() async throws {
    MLXRandom.seed(42)
    let rec = RecordingTelemetryReporter()
    let pipeline = try await makePipeline(rec: rec)

    // width=64, height=64 → latentHeight=8, latentWidth=8, channels=4
    // latent shape: [1, 8, 8, 4]
    let expectedShape = [1, 8, 8, 4]
    let request = DiffusionGenerationRequest(
      prompt: "shape test",
      width: 64, height: 64,
      steps: 4,
      guidanceScale: 1.0,
      seed: 2
    )
    _ = try await pipeline.generate(request: request, progress: { _ in })

    let events = await rec.events
    let completeStats = events.compactMap { e -> TuberiaTensorStat? in
      if case .denoiseStepComplete(_, _, _, _, let latentAfterStat, _, _) = e {
        return latentAfterStat
      } else {
        return nil
      }
    }

    #expect(completeStats.count == 4)
    for stat in completeStats {
      #expect(
        stat.shape == expectedShape,
        "latentAfterStat.shape \(stat.shape) != expected \(expectedShape)"
      )
    }
  }
}
