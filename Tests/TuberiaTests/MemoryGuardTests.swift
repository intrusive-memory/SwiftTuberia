import CoreGraphics
import Foundation
@preconcurrency import MLX
import SwiftAcervo
import Testing

@testable import Tuberia

// MARK: - Minimal Mocks (MemoryGuardTests-local)

private final class MemGuardEncoder: TextEncoder, @unchecked Sendable {
  struct Config: Sendable {
    let dim: Int
    let seqLen: Int
  }
  typealias Configuration = Config
  private(set) var isLoaded = false
  private var weights: ModuleParameters?
  var outputEmbeddingDim: Int { configuration.dim }
  var maxSequenceLength: Int { configuration.seqLen }
  var estimatedMemoryBytes: Int { 1_000_000 }  // 1 MB — contributes to peakMemoryBytes
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

private final class MemGuardBackbone: Backbone, @unchecked Sendable {
  struct Config: Sendable {
    let dim: Int
    let seqLen: Int
  }
  typealias Configuration = Config
  private(set) var isLoaded = false
  private var weights: ModuleParameters?
  var expectedConditioningDim: Int { configuration.dim }
  var outputLatentChannels: Int { 4 }
  var expectedMaxSequenceLength: Int { configuration.seqLen }
  var estimatedMemoryBytes: Int { 1_000_000 }  // 1 MB — contributes to peakMemoryBytes
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { weights }
  private let configuration: Config
  required init(configuration: Config) throws { self.configuration = configuration }
  func forward(_ input: BackboneInput) throws -> MLXArray { MLXArray.ones(input.latents.shape) }
  func apply(weights: ModuleParameters) throws {
    self.weights = weights
    isLoaded = true
  }
  func unload() {
    weights = nil
    isLoaded = false
  }
}

private final class MemGuardDecoder: Decoder, @unchecked Sendable {
  struct Config: Sendable {}
  typealias Configuration = Config
  private(set) var isLoaded = false
  private var weights: ModuleParameters?
  var expectedInputChannels: Int { 4 }
  var scalingFactor: Float { 0.18215 }
  var estimatedMemoryBytes: Int { 1_000_000 }  // 1 MB — contributes to peakMemoryBytes
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

private final class MemGuardScheduler: Scheduler, @unchecked Sendable {
  struct Config: Sendable {}
  typealias Configuration = Config
  required init(configuration: Config) {}
  func configure(steps: Int, startTimestep: Int?) -> SchedulerPlan {
    SchedulerPlan(timesteps: [], sigmas: [])
  }
  func step(output: MLXArray, timestep: Int, sample: MLXArray) -> MLXArray { sample }
  func addNoise(to sample: MLXArray, noise: MLXArray, at timestep: Int) -> MLXArray { sample }
  func reset() {}
}

private final class MemGuardRenderer: Renderer, @unchecked Sendable {
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
      throw PipelineError.renderingFailed(reason: "MemGuardRenderer: CGImage failed")
    }
    return .image(img)
  }
}

/// Minimal recipe for memory-guard tests. All component IDs are nil so WeightLoader
/// is never invoked — the memory gate fires (or doesn't) before any I/O.
private struct MemGuardRecipe: PipelineRecipe, Sendable {
  typealias Encoder = MemGuardEncoder
  typealias Sched = MemGuardScheduler
  typealias Back = MemGuardBackbone
  typealias Dec = MemGuardDecoder
  typealias Rend = MemGuardRenderer

  let encoderConfig = MemGuardEncoder.Config(dim: 64, seqLen: 8)
  let schedulerConfig = MemGuardScheduler.Config()
  let backboneConfig = MemGuardBackbone.Config(dim: 64, seqLen: 8)
  let decoderConfig = MemGuardDecoder.Config()
  let rendererConfig: Void = ()

  let supportsImageToImage = false
  let unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .emptyPrompt
  let allComponentIds: [String] = []  // No component IDs → no WeightLoader calls

  func quantizationFor(_ role: PipelineRole) -> QuantizationConfig { .asStored }
  func validate() throws {}
}

// MARK: - Tests

/// Verifies the memory gate wired into `loadModels(progress:)` by REQ-PIPE-02 (S4).
///
/// The gate calls `MemoryManager.shared.hardValidate(requiredBytes: peakMemoryBytes)`.
/// These tests inject a stub gate that simulates insufficient memory without querying
/// real hardware, then assert `PipelineError.insufficientMemory` is thrown.
///
/// The stub gate is injected via `pipeline.setMemoryGate(_:)` — the same seam pattern
/// used by `setComponentReadinessService(_:)` (REQ-PIPE-01, S3).
@Suite("MemoryGuard Tests", .serialized)
struct MemoryGuardTests {

  // MARK: - Helpers

  private typealias TestPipeline = DiffusionPipeline<
    MemGuardEncoder,
    MemGuardScheduler,
    MemGuardBackbone,
    MemGuardDecoder,
    MemGuardRenderer
  >

  private func makePipeline() throws -> TestPipeline {
    try TestPipeline(recipe: MemGuardRecipe())
  }

  // MARK: - Insufficient Memory Gate

  @Test("loadModels throws insufficientMemory when gate simulates exhausted memory")
  func throwsInsuffcientMemoryWhenBudgetExceeded() async throws {
    let pipeline = try makePipeline()

    // The pipeline's peakMemoryBytes = encoder (1 MB) + backbone (1 MB) + decoder (1 MB) = 3 MB.
    // Simulate a device with only 1 byte available — guaranteed to fail the gate.
    let simulatedAvailable: UInt64 = 1
    await pipeline.setMemoryGate { requiredBytes in
      guard simulatedAvailable >= requiredBytes else {
        throw PipelineError.insufficientMemory(
          required: requiredBytes,
          available: simulatedAvailable,
          component: "pipeline"
        )
      }
    }

    var caughtInsufficient = false
    do {
      try await pipeline.loadModels { _, _ in }
      Issue.record("Expected PipelineError.insufficientMemory but loadModels succeeded")
    } catch let error as PipelineError {
      switch error {
      case .insufficientMemory(let required, let available, _):
        // required must equal peakMemoryBytes (3 MB from the three 1 MB components).
        #expect(required == 3_000_000, "required should equal peakMemoryBytes (3 MB)")
        #expect(available == simulatedAvailable, "available should equal the stub value")
        caughtInsufficient = true
      default:
        Issue.record("Expected .insufficientMemory, got \(error)")
      }
    } catch {
      Issue.record("Expected PipelineError.insufficientMemory, got \(error)")
    }
    #expect(caughtInsufficient, "PipelineError.insufficientMemory must be thrown")
  }

  // MARK: - Sufficient Memory Gate

  @Test("loadModels does NOT throw insufficientMemory when gate simulates ample memory")
  func noThrowWhenBudgetSufficient() async throws {
    let pipeline = try makePipeline()

    // Simulate a device with effectively unlimited memory — gate must pass.
    await pipeline.setMemoryGate { _ in
      // No-op: memory is plentiful.
    }

    // loadModels will proceed past the memory gate. WeightLoader.load will fail
    // because there are no real weights, but that is a different error — NOT
    // PipelineError.insufficientMemory.
    do {
      try await pipeline.loadModels { _, _ in }
      // Success path: no component IDs → no loading attempted, completes cleanly.
    } catch let error as PipelineError {
      if case .insufficientMemory = error {
        Issue.record(
          "Gate was stubbed to pass; insufficientMemory must NOT be thrown. Got: \(error)")
      }
      // Other PipelineError cases are acceptable — the test only guards against spurious
      // insufficientMemory.
    } catch {
      // Non-PipelineError failures are also acceptable here.
    }
  }

  // MARK: - Gate Receives peakMemoryBytes

  @Test("loadModels passes peakMemoryBytes to the memory gate")
  func gateReceivesPeakMemoryBytes() async throws {
    let pipeline = try makePipeline()

    // peakMemoryBytes = encoder + backbone + decoder = 1 MB + 1 MB + 1 MB = 3 MB.
    let expectedPeak: UInt64 = 3_000_000

    // Use an actor-isolated box to safely capture the observed value from the
    // @Sendable gate closure without racing.
    let observed = ObservedRequiredBox()
    await pipeline.setMemoryGate { requiredBytes in
      await observed.set(requiredBytes)
      // Allow the load to proceed (memory sufficient).
    }

    do {
      try await pipeline.loadModels { _, _ in }
    } catch {
      // Failures from WeightLoader are expected when no real weights exist — ignore.
    }

    let observedRequired = await observed.value
    #expect(
      observedRequired == expectedPeak,
      "memoryGate must receive exactly peakMemoryBytes (\(expectedPeak)), got \(observedRequired)"
    )
  }
}

// MARK: - ObservedRequiredBox

/// Actor-isolated box for safely capturing a UInt64 value from a @Sendable closure.
private actor ObservedRequiredBox {
  private(set) var value: UInt64 = 0
  func set(_ v: UInt64) { value = v }
}
