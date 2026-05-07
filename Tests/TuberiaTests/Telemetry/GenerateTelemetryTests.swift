import CoreGraphics
import Foundation
@preconcurrency import MLX
import Testing

@testable import Tuberia

// MARK: - In-target stub components (TuberiaTests-only, no dependency on TuberiaGPUTests)
// Pattern mirrors DiffusionPipelineTelemetryTests.swift (Sortie 4).

private final class GenerateStubEncoder: TextEncoder, @unchecked Sendable {
  struct Config: Sendable {
    let dim: Int
    let seqLen: Int
  }
  typealias Configuration = Config
  // Always loaded — generate() guards require isLoaded == true.
  private(set) var isLoaded = true
  private var weights: ModuleParameters?
  var outputEmbeddingDim: Int { configuration.dim }
  var maxSequenceLength: Int { configuration.seqLen }
  var estimatedMemoryBytes: Int { 1_000 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { weights }
  private let configuration: Config
  required init(configuration: Config) throws { self.configuration = configuration }
  func encode(_ input: TextEncoderInput) throws -> TextEncoderOutput {
    TextEncoderOutput(
      embeddings: MLXArray.ones([1, input.maxLength, configuration.dim]),
      mask: MLXArray.ones([1, input.maxLength])
    )
  }
  func apply(weights: ModuleParameters) throws {
    self.weights = weights
    isLoaded = true
  }
  func unload() {
    weights = nil
    isLoaded = false
  }
}

/// Scheduler that returns exactly `steps` timesteps from configure(),
/// enabling precise denoiseStep count assertions.
private final class GenerateStubScheduler: Scheduler, @unchecked Sendable {
  struct Config: Sendable {}
  typealias Configuration = Config
  required init(configuration: Config) {}
  func configure(steps: Int, startTimestep: Int?) -> SchedulerPlan {
    // Return a plan with `steps` timesteps (one per requested step).
    // Sigmas are not used by the stub backbone, so fill with zeros.
    let ts = Array(0..<steps).reversed().map { $0 }
    return SchedulerPlan(timesteps: ts, sigmas: Array(repeating: 1.0, count: steps))
  }
  func step(output: MLXArray, timestep: Int, sample: MLXArray) -> MLXArray { sample }
  func addNoise(to sample: MLXArray, noise: MLXArray, at timestep: Int) -> MLXArray { sample }
  func reset() {}
}

private final class GenerateStubBackbone: Backbone, @unchecked Sendable {
  struct Config: Sendable {
    let dim: Int
    let seqLen: Int
  }
  typealias Configuration = Config
  // Always loaded — generate() guards require isLoaded == true.
  private(set) var isLoaded = true
  private var weights: ModuleParameters?
  var expectedConditioningDim: Int { configuration.dim }
  var outputLatentChannels: Int { 4 }
  var expectedMaxSequenceLength: Int { configuration.seqLen }
  var estimatedMemoryBytes: Int { 2_000 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { weights }
  private let configuration: Config
  required init(configuration: Config) throws { self.configuration = configuration }
  func forward(_ input: BackboneInput) throws -> MLXArray {
    // Return latents unchanged — no real diffusion work, no GPU allocation.
    MLXArray.zeros(input.latents.shape)
  }
  func apply(weights: ModuleParameters) throws {
    self.weights = weights
    isLoaded = true
  }
  func unload() {
    weights = nil
    isLoaded = false
  }
}

private final class GenerateStubDecoder: Decoder, @unchecked Sendable {
  struct Config: Sendable {}
  typealias Configuration = Config
  // Always loaded — generate() guards require isLoaded == true.
  private(set) var isLoaded = true
  private var weights: ModuleParameters?
  var expectedInputChannels: Int { 4 }
  var scalingFactor: Float { 0.18215 }
  var estimatedMemoryBytes: Int { 3_000 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { weights }
  required init(configuration: Config) throws {}
  func decode(_ latents: MLXArray) throws -> DecodedOutput {
    let shape = latents.shape
    let b = shape[0]
    let h = shape.count > 1 ? shape[1] * 8 : 8
    let w = shape.count > 2 ? shape[2] * 8 : 8
    return DecodedOutput(
      data: MLXArray.ones([b, h, w, 3]) * 0.5,
      metadata: ImageDecoderMetadata(scalingFactor: scalingFactor)
    )
  }
  func apply(weights: ModuleParameters) throws {
    self.weights = weights
    isLoaded = true
  }
  func unload() {
    weights = nil
    isLoaded = false
  }
}

private final class GenerateStubRenderer: Renderer, @unchecked Sendable {
  typealias Configuration = Void
  required init(configuration: Void) {}
  func render(_ input: DecodedOutput) throws -> RenderedOutput {
    var pixelData: [UInt8] = [128, 128, 128, 255]
    let space = CGColorSpaceCreateDeviceRGB()
    guard
      let ctx = CGContext(
        data: &pixelData, width: 1, height: 1,
        bitsPerComponent: 8, bytesPerRow: 4, space: space,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      ),
      let img = ctx.makeImage()
    else {
      throw PipelineError.renderingFailed(reason: "GenerateStubRenderer: CGImage failed")
    }
    return .image(img)
  }
}

/// Minimal CPU-only stub recipe for generate-flow telemetry tests.
/// `isLoaded` is pre-set to `true` on all three weighted components so
/// `generate(request:progress:)` passes its `isLoaded` guards without
/// a real `loadModels` call.
private struct GenerateTelemetryStubRecipe: PipelineRecipe, Sendable {
  typealias Encoder = GenerateStubEncoder
  typealias Sched = GenerateStubScheduler
  typealias Back = GenerateStubBackbone
  typealias Dec = GenerateStubDecoder
  typealias Rend = GenerateStubRenderer

  let encoderConfig = GenerateStubEncoder.Config(dim: 64, seqLen: 8)
  let schedulerConfig = GenerateStubScheduler.Config()
  let backboneConfig = GenerateStubBackbone.Config(dim: 64, seqLen: 8)
  let decoderConfig = GenerateStubDecoder.Config()
  let rendererConfig: Void = ()

  let supportsImageToImage = false
  let unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .emptyPrompt
  /// Empty component IDs: no WeightLoader calls, no file I/O.
  let allComponentIds: [String] = []

  func quantizationFor(_ role: PipelineRole) -> QuantizationConfig { .asStored }
  func validate() throws {}
}

// MARK: - Case-discrimination helpers (generate-flow events)

private func isGenerateStart(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .generateStart = e { return true }
  return false
}
private func isEncodeStart(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .encodeStart = e { return true }
  return false
}
private func isEncodeComplete(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .encodeComplete = e { return true }
  return false
}
private func isDenoiseStep(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .denoiseStep = e { return true }
  return false
}
private func isDecodeStart(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .decodeStart = e { return true }
  return false
}
private func isDecodeComplete(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .decodeComplete = e { return true }
  return false
}
private func isRenderStart(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .renderStart = e { return true }
  return false
}
private func isRenderComplete(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .renderComplete = e { return true }
  return false
}
private func isGenerateComplete(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .generateComplete = e { return true }
  return false
}

// MARK: - Test Suite

@Suite("Generate Telemetry Tests", .serialized)
struct GenerateTelemetryTests {

  init() { TestEnvironment.ensureAcervoAppGroup() }

  // MARK: - Pipeline type alias and factory

  private typealias TestPipeline = DiffusionPipeline<
    GenerateStubEncoder,
    GenerateStubScheduler,
    GenerateStubBackbone,
    GenerateStubDecoder,
    GenerateStubRenderer
  >

  private func makePipeline() throws -> TestPipeline {
    try TestPipeline(recipe: GenerateTelemetryStubRecipe())
  }

  /// Small step count keeps the test fast while giving a non-trivial denoiseStep sequence.
  private let requestSteps = 4

  private func makeRequest(steps: Int) -> DiffusionGenerationRequest {
    DiffusionGenerationRequest(
      prompt: "test prompt",
      width: 64,
      height: 64,
      steps: steps,
      guidanceScale: 1.0,  // <= 1.0 skips CFG — single backbone pass per step
      seed: 42
    )
  }

  // MARK: - Test 1: denoiseStep count equals request.steps

  @Test("denoiseStep event count equals request.steps")
  func denoiseStepCountEqualsRequestSteps() async throws {
    let pipeline = try makePipeline()
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)

    let request = makeRequest(steps: requestSteps)
    _ = try await pipeline.generate(request: request, progress: { _ in })

    let events = await mock.events
    let denoiseStepCount = events.filter(isDenoiseStep).count

    #expect(
      denoiseStepCount == requestSteps,
      "denoiseStep must fire exactly \(requestSteps) times (once per step); got \(denoiseStepCount)"
    )
  }

  // MARK: - Test 2: Canonical generate-flow ordering

  @Test("generate emits events in canonical flow order")
  func canonicalGenerateOrdering() async throws {
    let pipeline = try makePipeline()
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)

    let request = makeRequest(steps: requestSteps)
    _ = try await pipeline.generate(request: request, progress: { _ in })

    let events = await mock.events

    // Locate mandatory milestone events by index.
    let genStartIdx = try #require(
      events.firstIndex(where: isGenerateStart),
      "generateStart must be emitted"
    )
    let encStartIdx = try #require(
      events.firstIndex(where: isEncodeStart),
      "encodeStart must be emitted"
    )
    let encCompleteIdx = try #require(
      events.firstIndex(where: isEncodeComplete),
      "encodeComplete must be emitted"
    )
    let decStartIdx = try #require(
      events.firstIndex(where: isDecodeStart),
      "decodeStart must be emitted"
    )
    let decCompleteIdx = try #require(
      events.firstIndex(where: isDecodeComplete),
      "decodeComplete must be emitted"
    )
    let rendStartIdx = try #require(
      events.firstIndex(where: isRenderStart),
      "renderStart must be emitted"
    )
    let rendCompleteIdx = try #require(
      events.firstIndex(where: isRenderComplete),
      "renderComplete must be emitted"
    )
    let genCompleteIdx = try #require(
      events.firstIndex(where: isGenerateComplete),
      "generateComplete must be emitted"
    )

    // Milestone ordering.
    #expect(genStartIdx < encStartIdx,    "generateStart must precede encodeStart")
    #expect(encStartIdx < encCompleteIdx, "encodeStart must precede encodeComplete")
    #expect(encCompleteIdx < decStartIdx, "encodeComplete must precede decodeStart")
    #expect(decStartIdx < decCompleteIdx, "decodeStart must precede decodeComplete")
    #expect(decCompleteIdx < rendStartIdx, "decodeComplete must precede renderStart")
    #expect(rendStartIdx < rendCompleteIdx, "renderStart must precede renderComplete")
    #expect(rendCompleteIdx < genCompleteIdx, "renderComplete must precede generateComplete")

    // All denoiseStep events must be bracketed between encodeComplete and decodeStart.
    let denoiseIndices = events.indices.filter { isDenoiseStep(events[$0]) }
    #expect(!denoiseIndices.isEmpty, "At least one denoiseStep must be emitted")

    if let firstDenoise = denoiseIndices.first, let lastDenoise = denoiseIndices.last {
      #expect(
        encCompleteIdx < firstDenoise,
        "First denoiseStep must follow encodeComplete"
      )
      #expect(
        lastDenoise < decStartIdx,
        "Last denoiseStep must precede decodeStart"
      )
    }
  }

  // MARK: - Test 3: generateComplete.totalDurationSeconds matches result.duration

  @Test("generateComplete.totalDurationSeconds matches DiffusionGenerationResult.duration")
  func generateCompleteDurationMatchesResult() async throws {
    let pipeline = try makePipeline()
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)

    let request = makeRequest(steps: requestSteps)
    let result = try await pipeline.generate(request: request, progress: { _ in })

    let events = await mock.events

    // Extract the generateComplete event.
    let completeEvent = try #require(
      events.first(where: isGenerateComplete),
      "generateComplete must be emitted"
    )

    guard case .generateComplete(let emittedDuration, _, _) = completeEvent else {
      Issue.record("Expected .generateComplete; got \(completeEvent)")
      return
    }

    // Both the event and the result read from the same `duration` local in DiffusionPipeline,
    // so they should be bit-equal. Use a tolerance of 0.001 seconds to be defensive.
    let delta = abs(emittedDuration - result.duration)
    #expect(
      delta < 0.001,
      "generateComplete.totalDurationSeconds (\(emittedDuration)) must match result.duration (\(result.duration)); delta=\(delta)"
    )
  }
}
