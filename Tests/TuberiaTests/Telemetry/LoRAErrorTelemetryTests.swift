import CoreGraphics
import Foundation
@preconcurrency import MLX
import Testing

@testable import Tuberia

// MARK: - Throw controllers
// Each controller is a `final class` (reference type) shared between the test and the
// stub component. The test holds a strong reference; the stub's Config also holds one.
// Mutating `shouldThrow` on the test side is visible to the stub because they share
// the same object — no actor hop required.

private final class EncoderThrowController: @unchecked Sendable {
  var shouldThrow: Bool = false
}

private final class DecoderThrowController: @unchecked Sendable {
  var shouldThrow: Bool = false
}

private final class RendererThrowController: @unchecked Sendable {
  var shouldThrow: Bool = false
}

// MARK: - In-target stub components (TuberiaTests-only, no dependency on TuberiaGPUTests)
// Pattern mirrors GenerateTelemetryTests.swift (Sortie 7).
// Each component reads from a shared controller so the test can inject throws
// without needing actor-isolated access to the pipeline's internal components.

private final class LoRAStubEncoder: TextEncoder, @unchecked Sendable {
  struct Config: Sendable {
    let dim: Int
    let seqLen: Int
    let throwController: EncoderThrowController
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
    if configuration.throwController.shouldThrow {
      throw PipelineError.encodingFailed(reason: "LoRAStubEncoder: injected throw")
    }
    return TextEncoderOutput(
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

private final class LoRAStubScheduler: Scheduler, @unchecked Sendable {
  struct Config: Sendable {}
  typealias Configuration = Config
  required init(configuration: Config) {}
  func configure(steps: Int, startTimestep: Int?) -> SchedulerPlan {
    // Return exactly `steps` timesteps so denoiseStep fires the right number of times.
    let ts = Array(0..<steps).reversed().map { $0 }
    return SchedulerPlan(timesteps: ts, sigmas: Array(repeating: 1.0, count: steps))
  }
  func step(output: MLXArray, timestep: Int, sample: MLXArray) -> MLXArray { sample }
  func addNoise(to sample: MLXArray, noise: MLXArray, at timestep: Int) -> MLXArray { sample }
  func reset() {}
}

private final class LoRAStubBackbone: Backbone, @unchecked Sendable {
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

private final class LoRAStubDecoder: Decoder, @unchecked Sendable {
  struct Config: Sendable {
    let throwController: DecoderThrowController
  }
  typealias Configuration = Config
  // Always loaded — generate() guards require isLoaded == true.
  private(set) var isLoaded = true
  private var weights: ModuleParameters?
  var expectedInputChannels: Int { 4 }
  var scalingFactor: Float { 0.18215 }
  var estimatedMemoryBytes: Int { 3_000 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { weights }
  private let configuration: Config
  required init(configuration: Config) throws { self.configuration = configuration }
  func decode(_ latents: MLXArray) throws -> DecodedOutput {
    if configuration.throwController.shouldThrow {
      throw PipelineError.decodingFailed(reason: "LoRAStubDecoder: injected throw")
    }
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

private final class LoRAStubRenderer: Renderer, @unchecked Sendable {
  struct Config: Sendable {
    let throwController: RendererThrowController
  }
  typealias Configuration = Config
  private let configuration: Config
  required init(configuration: Config) { self.configuration = configuration }
  func render(_ input: DecodedOutput) throws -> RenderedOutput {
    if configuration.throwController.shouldThrow {
      throw PipelineError.renderingFailed(reason: "LoRAStubRenderer: injected throw")
    }
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
      throw PipelineError.renderingFailed(reason: "LoRAStubRenderer: CGImage failed")
    }
    return .image(img)
  }
}

/// Minimal CPU-only stub recipe for LoRA / errorThrown telemetry tests.
/// `isLoaded` is pre-set to `true` on all three weighted components so
/// `generate(request:progress:)` passes its `isLoaded` guards without
/// a real `loadModels` call.
///
/// Throw controllers are shared references: each test creates its own set
/// of controllers, wires them into the recipe, and then flips the relevant
/// `shouldThrow` flag to inject an error for that test variant.
private struct LoRAErrorStubRecipe: PipelineRecipe, Sendable {
  typealias Encoder = LoRAStubEncoder
  typealias Sched = LoRAStubScheduler
  typealias Back = LoRAStubBackbone
  typealias Dec = LoRAStubDecoder
  typealias Rend = LoRAStubRenderer

  let encoderConfig: LoRAStubEncoder.Config
  let schedulerConfig = LoRAStubScheduler.Config()
  let backboneConfig = LoRAStubBackbone.Config(dim: 64, seqLen: 8)
  let decoderConfig: LoRAStubDecoder.Config
  let rendererConfig: LoRAStubRenderer.Config

  let supportsImageToImage = false
  let unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .emptyPrompt
  /// Empty component IDs: no WeightLoader calls, no file I/O.
  let allComponentIds: [String] = []

  func quantizationFor(_ role: PipelineRole) -> QuantizationConfig { .asStored }
  func validate() throws {}
}

// MARK: - Case-discrimination helpers (LoRA and error events)

private func isLoraLoadStart(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .loraLoadStart = e { return true }
  return false
}

private func isLoraMerge(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .loraMerge = e { return true }
  return false
}

private func isLoraUnmerge(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .loraUnmerge = e { return true }
  return false
}

private func isErrorThrown(_ e: TuberiaTelemetryEvent) -> Bool {
  if case .errorThrown = e { return true }
  return false
}

private func isErrorThrownWithPhase(_ targetPhase: String) -> (TuberiaTelemetryEvent) -> Bool {
  return { event in
    if case .errorThrown(let phase, _) = event { return phase == targetPhase }
    return false
  }
}

// MARK: - Test Suite

@Suite("LoRA / errorThrown Telemetry Tests", .serialized)
struct LoRAErrorTelemetryTests {

  init() { TestEnvironment.ensureAcervoAppGroup() }

  // MARK: - Pipeline factory

  private typealias TestPipeline = DiffusionPipeline<
    LoRAStubEncoder,
    LoRAStubScheduler,
    LoRAStubBackbone,
    LoRAStubDecoder,
    LoRAStubRenderer
  >

  private struct Controllers: Sendable {
    let encoder = EncoderThrowController()
    let decoder = DecoderThrowController()
    let renderer = RendererThrowController()
  }

  private func makePipeline(controllers: Controllers) throws -> TestPipeline {
    let recipe = LoRAErrorStubRecipe(
      encoderConfig: LoRAStubEncoder.Config(
        dim: 64, seqLen: 8, throwController: controllers.encoder),
      decoderConfig: LoRAStubDecoder.Config(throwController: controllers.decoder),
      rendererConfig: LoRAStubRenderer.Config(throwController: controllers.renderer)
    )
    return try TestPipeline(recipe: recipe)
  }

  private func makeRequest(
    steps: Int = 2,
    loRA: LoRAConfig? = nil
  ) -> DiffusionGenerationRequest {
    DiffusionGenerationRequest(
      prompt: "test prompt",
      width: 64,
      height: 64,
      steps: steps,
      guidanceScale: 1.0,  // <= 1.0 skips CFG — single backbone pass per step
      seed: 42,
      loRA: loRA
    )
  }

  // MARK: - Test 1: LoRA events fire when request has loRA
  //
  // LoRA design note: There is no injection seam for LoRALoader. Providing a non-existent
  // localPath causes LoRALoader.loadAdapterWeights → WeightLoader.loadFromPath to throw
  // PipelineError.weightLoadingFailed (or similar I/O error) AFTER `loraLoadStart` has
  // already been emitted. This means:
  //   - `loraLoadStart` MUST fire (emitted synchronously before the throwing async call).
  //   - `loraMerge` and `loraUnmerge` MUST NOT fire (they require a successful weight load).
  //   - The caller observes a PipelineError (error propagates; not swallowed).
  // This is the documented fallback from the plan: "assert that events fire BEFORE the loader
  // fails (acceptable)." The loraLoadStart assertion is the strongest guarantee available
  // without a real LoRA adapter file or a LoRALoader injection seam.

  @Test("loraLoadStart fires when request.loRA is set; loraMerge and loraUnmerge absent on load failure")
  func loRAEventsFireWhenRequestHasLoRA() async throws {
    let controllers = Controllers()
    let pipeline = try makePipeline(controllers: controllers)
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)

    // Use a non-existent local path. LoRALoader.loadFromPath will fail, but loraLoadStart
    // is emitted BEFORE LoRALoader.loadAdapterWeights is called — so we can assert it fires.
    let loraConfig = LoRAConfig(
      localPath: "/nonexistent/stub-lora.safetensors",
      scale: 0.8
    )
    let request = makeRequest(loRA: loraConfig)

    // The call must throw because the fake LoRA path doesn't exist.
    var didThrow = false
    do {
      _ = try await pipeline.generate(request: request, progress: { _ in })
    } catch is PipelineError {
      didThrow = true
    } catch {
      Issue.record("Expected PipelineError from LoRA load failure; got \(error)")
    }
    #expect(didThrow, "generate must throw when LoRA localPath is non-existent")

    let events = await mock.events

    // loraLoadStart must have fired before the failure.
    let loraLoadStartEvents = events.filter(isLoraLoadStart)
    #expect(
      loraLoadStartEvents.count >= 1,
      "loraLoadStart must fire even when LoRALoader subsequently fails (emitted before the load call)"
    )

    // Verify the loraLoadStart event carries the correct payload.
    if let event = loraLoadStartEvents.first,
      case .loraLoadStart(let adapterPath, let scale, _) = event
    {
      #expect(
        adapterPath == "/nonexistent/stub-lora.safetensors",
        "loraLoadStart.adapterPath must match LoRAConfig.localPath"
      )
      #expect(scale == 0.8, "loraLoadStart.scale must match LoRAConfig.scale")
    } else {
      Issue.record("Expected .loraLoadStart event; found none or wrong type")
    }

    // loraMerge and loraUnmerge must NOT fire — the weight load failed before they could.
    let loraMergeCount = events.filter(isLoraMerge).count
    let loraUnmergeCount = events.filter(isLoraUnmerge).count
    #expect(loraMergeCount == 0, "loraMerge must not fire when LoRA weight load fails")
    #expect(loraUnmergeCount == 0, "loraUnmerge must not fire when LoRA weight load fails")
  }

  // MARK: - Test 2: LoRA events absent when request.loRA is nil

  @Test("loraLoadStart / loraMerge / loraUnmerge are all absent when request.loRA is nil")
  func loRAEventsAbsentWhenRequestNil() async throws {
    let controllers = Controllers()
    let pipeline = try makePipeline(controllers: controllers)
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)

    // Request with no LoRA — must complete successfully.
    let request = makeRequest(loRA: nil)
    _ = try await pipeline.generate(request: request, progress: { _ in })

    let events = await mock.events

    let loraLoadStartCount = events.filter(isLoraLoadStart).count
    let loraMergeCount = events.filter(isLoraMerge).count
    let loraUnmergeCount = events.filter(isLoraUnmerge).count

    #expect(
      loraLoadStartCount == 0,
      "loraLoadStart must not fire when request.loRA is nil; got \(loraLoadStartCount)"
    )
    #expect(
      loraMergeCount == 0,
      "loraMerge must not fire when request.loRA is nil; got \(loraMergeCount)"
    )
    #expect(
      loraUnmergeCount == 0,
      "loraUnmerge must not fire when request.loRA is nil; got \(loraUnmergeCount)"
    )
  }

  // MARK: - Test 3: Encoder throw emits errorThrown(phase: "generate.encode")
  //
  // Throw-injection pattern: flip `controllers.encoder.shouldThrow = true` before
  // calling `generate(...)`. The stub's Config holds a reference to the shared
  // EncoderThrowController, so the pipeline's internal encoder instance sees the change.
  //
  // DiffusionPipeline catches the throw from `encoder.encode(_:)`, wraps it as
  // `PipelineError.encodingFailed`, emits `errorThrown(phase: "generate.encode", …)`,
  // then rethrows — so the caller still receives the original error kind.

  @Test("encoder throw emits errorThrown(phase: \"generate.encode\") and propagates to caller")
  func encodeErrorEmitsErrorThrown() async throws {
    let controllers = Controllers()
    let pipeline = try makePipeline(controllers: controllers)
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)

    // Inject throw into the encoder via the shared controller.
    controllers.encoder.shouldThrow = true

    var didCatchEncodingError = false
    do {
      _ = try await pipeline.generate(request: makeRequest(), progress: { _ in })
      Issue.record("Expected PipelineError.encodingFailed but generate succeeded")
    } catch let error as PipelineError {
      if case .encodingFailed = error {
        didCatchEncodingError = true
      } else {
        Issue.record("Expected .encodingFailed; got \(error)")
      }
    } catch {
      Issue.record("Expected PipelineError; got \(error)")
    }

    #expect(
      didCatchEncodingError,
      "Caller must observe the rethrown PipelineError.encodingFailed"
    )

    let events = await mock.events

    // errorThrown with phase "generate.encode" must be present.
    let encodeErrorEvents = events.filter(isErrorThrownWithPhase("generate.encode"))
    #expect(
      encodeErrorEvents.count >= 1,
      "errorThrown(phase: \"generate.encode\") must be emitted when encoder throws"
    )
  }

  // MARK: - Test 4: Decoder throw emits errorThrown(phase: "generate.decode")

  @Test("decoder throw emits errorThrown(phase: \"generate.decode\") and propagates to caller")
  func decodeErrorEmitsErrorThrown() async throws {
    let controllers = Controllers()
    let pipeline = try makePipeline(controllers: controllers)
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)

    // Inject throw into the decoder via the shared controller.
    controllers.decoder.shouldThrow = true

    var didCatchDecodingError = false
    do {
      _ = try await pipeline.generate(request: makeRequest(), progress: { _ in })
      Issue.record("Expected PipelineError.decodingFailed but generate succeeded")
    } catch let error as PipelineError {
      if case .decodingFailed = error {
        didCatchDecodingError = true
      } else {
        Issue.record("Expected .decodingFailed; got \(error)")
      }
    } catch {
      Issue.record("Expected PipelineError; got \(error)")
    }

    #expect(
      didCatchDecodingError,
      "Caller must observe the rethrown PipelineError.decodingFailed"
    )

    let events = await mock.events

    // errorThrown with phase "generate.decode" must be present.
    let decodeErrorEvents = events.filter(isErrorThrownWithPhase("generate.decode"))
    #expect(
      decodeErrorEvents.count >= 1,
      "errorThrown(phase: \"generate.decode\") must be emitted when decoder throws"
    )
  }

  // MARK: - Test 5: Renderer throw emits errorThrown(phase: "generate.render")

  @Test("renderer throw emits errorThrown(phase: \"generate.render\") and propagates to caller")
  func renderErrorEmitsErrorThrown() async throws {
    let controllers = Controllers()
    let pipeline = try makePipeline(controllers: controllers)
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)

    // Inject throw into the renderer via the shared controller.
    controllers.renderer.shouldThrow = true

    var didCatchRenderingError = false
    do {
      _ = try await pipeline.generate(request: makeRequest(), progress: { _ in })
      Issue.record("Expected PipelineError.renderingFailed but generate succeeded")
    } catch let error as PipelineError {
      if case .renderingFailed = error {
        didCatchRenderingError = true
      } else {
        Issue.record("Expected .renderingFailed; got \(error)")
      }
    } catch {
      Issue.record("Expected PipelineError; got \(error)")
    }

    #expect(
      didCatchRenderingError,
      "Caller must observe the rethrown PipelineError.renderingFailed"
    )

    let events = await mock.events

    // errorThrown with phase "generate.render" must be present.
    let renderErrorEvents = events.filter(isErrorThrownWithPhase("generate.render"))
    #expect(
      renderErrorEvents.count >= 1,
      "errorThrown(phase: \"generate.render\") must be emitted when renderer throws"
    )
  }
}
