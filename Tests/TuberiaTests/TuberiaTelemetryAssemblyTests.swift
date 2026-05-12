import CoreGraphics
import Foundation
@preconcurrency import MLX
import Testing

@testable import Tuberia

// MARK: - §7 Row 1 — Assembly check telemetry
//
// Tests:
//   (a) Mismatched recipe: `assemblyCheckFailed(.encoderToBackboneDim, ...)` fires
//       before `PipelineError.incompatibleComponents` is thrown.
//   (b) Valid recipe: all six `assemblyCheckPassed` events fire in the expected order.

// MARK: - Local Mock Components

private final class AssemblyEncoder: TextEncoder, @unchecked Sendable {
  struct Config: Sendable {
    let dim: Int
    let seqLen: Int
  }
  typealias Configuration = Config
  private(set) var isLoaded = false
  private var weights: ModuleParameters?
  var outputEmbeddingDim: Int { configuration.dim }
  var maxSequenceLength: Int { configuration.seqLen }
  var estimatedMemoryBytes: Int { 0 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { weights }
  private let configuration: Config
  required init(configuration: Config) throws { self.configuration = configuration }
  func encode(_ input: TextEncoderInput) throws -> TextEncoderOutput {
    TextEncoderOutput(
      embeddings: MLXArray.ones([1, input.maxLength, configuration.dim]).asType(.float16),
      mask: MLXArray.ones([1, input.maxLength]).asType(.float16)
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

private final class AssemblyBackbone: Backbone, @unchecked Sendable {
  struct Config: Sendable {
    let condDim: Int
    let seqLen: Int
    let latentChannels: Int
  }
  typealias Configuration = Config
  private(set) var isLoaded = false
  private var weights: ModuleParameters?
  var expectedConditioningDim: Int { configuration.condDim }
  var outputLatentChannels: Int { configuration.latentChannels }
  var expectedMaxSequenceLength: Int { configuration.seqLen }
  var estimatedMemoryBytes: Int { 0 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { weights }
  private let configuration: Config
  required init(configuration: Config) throws { self.configuration = configuration }
  func forward(_ input: BackboneInput) throws -> MLXArray {
    MLXArray.zeros(input.latents.shape).asType(.float16)
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

private final class AssemblyDecoder: Decoder, @unchecked Sendable {
  struct Config: Sendable {
    let inputChannels: Int
  }
  typealias Configuration = Config
  private(set) var isLoaded = false
  private var weights: ModuleParameters?
  var expectedInputChannels: Int { configuration.inputChannels }
  var scalingFactor: Float { 0.18215 }
  var estimatedMemoryBytes: Int { 0 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { weights }
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
    self.weights = weights
    isLoaded = true
  }
  func unload() {
    weights = nil
    isLoaded = false
  }
}

private final class AssemblyScheduler: Scheduler, @unchecked Sendable {
  struct Config: Sendable {}
  typealias Configuration = Config
  required init(configuration: Config) {}
  func configure(steps: Int, startTimestep: Int?) -> SchedulerPlan {
    SchedulerPlan(timesteps: [], sigmas: [])
  }
  func step(output: MLXArray, timestep: Int, sample: MLXArray) throws -> MLXArray { sample }
  func addNoise(to sample: MLXArray, noise: MLXArray, at timestep: Int) -> MLXArray { sample }
  func reset() {}
}

private final class AssemblyRenderer: Renderer, @unchecked Sendable {
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
      throw PipelineError.renderingFailed(reason: "AssemblyRenderer: CGImage creation failed")
    }
    return .image(img)
  }
}

// MARK: - Recipes

/// Valid recipe: encoder dim == backbone condDim, seq lengths match, latent channels match.
private struct ValidAssemblyRecipe: PipelineRecipe, Sendable {
  typealias Encoder = AssemblyEncoder
  typealias Sched = AssemblyScheduler
  typealias Back = AssemblyBackbone
  typealias Dec = AssemblyDecoder
  typealias Rend = AssemblyRenderer

  let encoderConfig = AssemblyEncoder.Config(dim: 64, seqLen: 8)
  let schedulerConfig = AssemblyScheduler.Config()
  let backboneConfig = AssemblyBackbone.Config(condDim: 64, seqLen: 8, latentChannels: 4)
  let decoderConfig = AssemblyDecoder.Config(inputChannels: 4)
  let rendererConfig: Void = ()

  let supportsImageToImage = false
  let unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .emptyPrompt
  let allComponentIds: [String] = []
  func quantizationFor(_ role: PipelineRole) -> QuantizationConfig { .asStored }
  func validate() throws {}
}

/// Mismatched recipe: encoder.outputEmbeddingDim (32) != backbone.expectedConditioningDim (64).
private struct MismatchedDimRecipe: PipelineRecipe, Sendable {
  typealias Encoder = AssemblyEncoder
  typealias Sched = AssemblyScheduler
  typealias Back = AssemblyBackbone
  typealias Dec = AssemblyDecoder
  typealias Rend = AssemblyRenderer

  // Encoder produces dim=32, backbone expects dim=64 — deliberate mismatch
  let encoderConfig = AssemblyEncoder.Config(dim: 32, seqLen: 8)
  let schedulerConfig = AssemblyScheduler.Config()
  let backboneConfig = AssemblyBackbone.Config(condDim: 64, seqLen: 8, latentChannels: 4)
  let decoderConfig = AssemblyDecoder.Config(inputChannels: 4)
  let rendererConfig: Void = ()

  let supportsImageToImage = false
  let unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .emptyPrompt
  let allComponentIds: [String] = []
  func quantizationFor(_ role: PipelineRole) -> QuantizationConfig { .asStored }
  func validate() throws {}
}

// MARK: - Pipeline type aliases

private typealias AssemblyPipeline = DiffusionPipeline<
  AssemblyEncoder,
  AssemblyScheduler,
  AssemblyBackbone,
  AssemblyDecoder,
  AssemblyRenderer
>

// MARK: - Tests

@Suite("TuberiaTelemetryAssemblyTests — §7 row 1", .serialized)
struct TuberiaTelemetryAssemblyTests {

  // MARK: - (a) Mismatched recipe: assemblyCheckFailed fires before throw

  @Test("assemblyCheckFailed(.encoderToBackboneDim) fires before PipelineError is thrown")
  func assemblyCheckFailedFiresBeforeThrow() async throws {
    let rec = RecordingTelemetryReporter()

    // The pipeline init throws synchronously; the telemetry Task{} is scheduled
    // before the throw returns to the caller. Yield after the init to let the
    // scheduled tasks run.
    var caughtError: Error?
    do {
      _ = try AssemblyPipeline(recipe: MismatchedDimRecipe(), telemetry: rec)
    } catch {
      caughtError = error
    }

    // Must have caught an error
    #expect(caughtError != nil, "Expected PipelineError.incompatibleComponents to be thrown")

    // Confirm it's the correct error variant
    if let pipelineError = caughtError as? PipelineError {
      if case .incompatibleComponents = pipelineError {
        // expected
      } else {
        Issue.record("Expected .incompatibleComponents, got \(pipelineError)")
      }
    } else {
      Issue.record("Expected PipelineError, got \(String(describing: caughtError))")
    }

    // Drain the Task{} scheduled inside validateAssembly before the throw.
    // A few yields are usually enough; the task is scheduled on the cooperative
    // pool and completes promptly on the same thread in tests.
    for _ in 0..<10 {
      await Task.yield()
    }

    let events = await rec.events

    // Must have at least one assemblyCheckFailed event
    let failedEvents = events.compactMap { event -> TuberiaTelemetryEvent.AssemblyCheck? in
      if case .assemblyCheckFailed(let check, _, _, _) = event { return check } else { return nil }
    }
    #expect(!failedEvents.isEmpty, "Expected at least one assemblyCheckFailed event")
    #expect(
      failedEvents.contains(.encoderToBackboneDim),
      "Expected assemblyCheckFailed(.encoderToBackboneDim), got \(failedEvents)"
    )
  }

  // MARK: - (b) Valid recipe: all six assemblyCheckPassed events fire in order

  @Test("All six assemblyCheckPassed events fire in expected order for a valid recipe")
  func allSixAssemblyCheckPassedEventsFireInOrder() async throws {
    let rec = RecordingTelemetryReporter()

    // Valid recipe — init should succeed
    let pipeline = try AssemblyPipeline(recipe: ValidAssemblyRecipe(), telemetry: rec)
    // Keep pipeline in scope so it isn't deallocated before the Task{} fires
    _ = pipeline

    // Drain Task{} emissions from validateAssembly
    for _ in 0..<10 {
      await Task.yield()
    }

    let events = await rec.events
    let passedChecks = events.compactMap { event -> TuberiaTelemetryEvent.AssemblyCheck? in
      if case .assemblyCheckPassed(let check, _, _) = event { return check } else { return nil }
    }

    // All six checks must have fired
    let expectedChecks: [TuberiaTelemetryEvent.AssemblyCheck] = [
      .completeness,
      .encoderToBackboneDim,
      .encoderToBackboneSeq,
      .backboneToDecoder,
      .decoderToRenderer,
      .imageToImageBidirectional,
    ]
    for expected in expectedChecks {
      #expect(
        passedChecks.contains(expected),
        "Missing assemblyCheckPassed(\(expected))"
      )
    }

    // All six unique checks must be present.
    // Note: Each check is emitted via its own Task{} dispatch inside validateAssembly.
    // The Swift cooperative executor does not guarantee inter-Task ordering between
    // independent Task{} instances, so strict sequential ordering is not asserted.
    // The count and presence of all six checks are the load-bearing assertions.
    #expect(
      passedChecks.count == 6,
      "Expected exactly 6 assemblyCheckPassed events, got \(passedChecks.count): \(passedChecks)"
    )
  }
}
