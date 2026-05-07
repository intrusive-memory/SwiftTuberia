import CoreGraphics
import Foundation
@preconcurrency import MLX
import Testing

@testable import Tuberia

// MARK: - In-target stub components (CPU-only, no MLX GPU work)
// Pattern mirrors GenerateTelemetryTests.swift (Sortie 7).
// Components are pre-loaded (isLoaded = true) so we can skip the loadModels
// file-I/O path and focus purely on the generate + unload overhead.

private final class OverheadStubEncoder: TextEncoder, @unchecked Sendable {
  struct Config: Sendable {
    let dim: Int
    let seqLen: Int
  }
  typealias Configuration = Config
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

private final class OverheadStubScheduler: Scheduler, @unchecked Sendable {
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

private final class OverheadStubBackbone: Backbone, @unchecked Sendable {
  struct Config: Sendable {
    let dim: Int
    let seqLen: Int
  }
  typealias Configuration = Config
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

private final class OverheadStubDecoder: Decoder, @unchecked Sendable {
  struct Config: Sendable {}
  typealias Configuration = Config
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

private final class OverheadStubRenderer: Renderer, @unchecked Sendable {
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
      throw PipelineError.renderingFailed(reason: "OverheadStubRenderer: CGImage failed")
    }
    return .image(img)
  }
}

/// Minimal CPU-only stub recipe for overhead timing tests.
/// Uses 4 denoising steps to keep each trial fast.
/// Empty `allComponentIds` → no WeightLoader calls → loadModels completes immediately.
private struct OverheadStubRecipe: PipelineRecipe, Sendable {
  typealias Encoder = OverheadStubEncoder
  typealias Sched = OverheadStubScheduler
  typealias Back = OverheadStubBackbone
  typealias Dec = OverheadStubDecoder
  typealias Rend = OverheadStubRenderer

  let encoderConfig = OverheadStubEncoder.Config(dim: 64, seqLen: 8)
  let schedulerConfig = OverheadStubScheduler.Config()
  let backboneConfig = OverheadStubBackbone.Config(dim: 64, seqLen: 8)
  let decoderConfig = OverheadStubDecoder.Config()
  let rendererConfig: Void = ()

  let supportsImageToImage = false
  let unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .emptyPrompt
  /// Empty component IDs: no WeightLoader calls, no file I/O.
  let allComponentIds: [String] = []

  func quantizationFor(_ role: PipelineRole) -> QuantizationConfig { .asStored }
  func validate() throws {}
}

// MARK: - Test Suite
//
// GATE: This suite uses a two-layer gate to prevent accidental execution:
//
//   Layer 1 — compile-time (#if): The suite body is only compiled when
//     `TUBERIA_TELEMETRY_OVERHEAD_ENABLED` is defined in
//     `SWIFT_ACTIVE_COMPILATION_CONDITIONS`. This flag is set by
//     `TelemetryOverheadEnabled.xcconfig`, which is passed to xcodebuild
//     only by the `make test-telemetry-overhead` Makefile target.
//     When the flag is absent (all normal builds, `make test`, CI), the
//     struct compiles to a no-op empty shell — Swift Testing discovers no
//     test methods and the suite is invisible in the report.
//
//   Layer 2 — belt-and-suspenders `.disabled` trait: Even if the xcconfig is
//     somehow applied unintentionally, the suite requires the
//     TUBERIA_TELEMETRY_OVERHEAD env var to equal "1" at runtime.
//     NOTE: macOS SPM test bundles do not propagate host env vars through
//     xcodebuild (TEST_RUNNER_ prefix does not work for SPM targets on
//     macOS). The xcconfig compile-time gate is therefore the primary
//     mechanism; the .disabled(if:) here serves as belt-and-suspenders
//     for any future environment where propagation does work.
//
// INVOCATION: Run manually before each release via `make test-telemetry-overhead`.
// Intentionally excluded from `make test` — perf tests are noisy on CI.

#if TUBERIA_TELEMETRY_OVERHEAD_ENABLED
@Suite("Telemetry Overhead Tests", .serialized)
struct TelemetryOverheadTests {

  init() { TestEnvironment.ensureAcervoAppGroup() }

  // MARK: - Pipeline type aliases

  private typealias OverheadPipeline = DiffusionPipeline<
    OverheadStubEncoder,
    OverheadStubScheduler,
    OverheadStubBackbone,
    OverheadStubDecoder,
    OverheadStubRenderer
  >

  private func makePipeline() throws -> OverheadPipeline {
    try OverheadPipeline(recipe: OverheadStubRecipe())
  }

  /// 4 denoising steps — fast enough for 10 trials per config but enough to
  /// exercise the denoiseStep hot path meaningfully.
  private let denoiseSteps = 4

  private func makeRequest() -> DiffusionGenerationRequest {
    DiffusionGenerationRequest(
      prompt: "overhead test",
      width: 64,
      height: 64,
      steps: denoiseSteps,
      guidanceScale: 1.0,  // <= 1.0 skips CFG — single backbone pass per step
      seed: 0
    )
  }

  /// Run one complete load + generate + unload cycle and return wall-clock seconds.
  private func runOneCycle(reporter: (any TuberiaTelemetryReporter)?) async throws -> Double {
    let pipeline = try makePipeline()
    await pipeline.setTelemetry(reporter)

    let start = ContinuousClock.now
    try await pipeline.loadModels(progress: { _, _ in })
    _ = try await pipeline.generate(request: makeRequest(), progress: { _ in })
    await pipeline.unloadModels()
    let elapsed = ContinuousClock.now - start

    return elapsed.timeInterval
  }

  // MARK: - Overhead assertion

  /// Asserts that `NoopTuberiaTelemetryReporter` adds less than 10% wall-clock
  /// overhead compared to a nil reporter over 10 alternating trials.
  ///
  /// Trial order: nil, Noop, nil, Noop, … (alternating to neutralise warm-up bias).
  ///
  /// Threshold: 10% relative difference (noopMean < nilMean × 1.10).
  /// REQUIREMENTS §6 states a ≤1% project goal; §8.2 codifies 10% as the assertion
  /// threshold because Swift Testing parallel scheduling on macos-26 is noisy and
  /// a flaky 1% guard would block releases on noise.
  @Test("NoopTuberiaTelemetryReporter overhead is below 10% vs nil reporter")
  func telemetryOverheadIsBelow10Percent() async throws {
    let trialCount = 10
    var nilTimes: [Double] = []
    var noopTimes: [Double] = []

    let noop = NoopTuberiaTelemetryReporter()

    // Alternating trial order: nil, Noop, nil, Noop, …
    for i in 0..<trialCount {
      if i % 2 == 0 {
        let t = try await runOneCycle(reporter: nil)
        nilTimes.append(t)
      } else {
        let t = try await runOneCycle(reporter: noop)
        noopTimes.append(t)
      }
    }

    let nilMean = nilTimes.reduce(0, +) / Double(nilTimes.count)
    let noopMean = noopTimes.reduce(0, +) / Double(noopTimes.count)

    let deltaFraction = (noopMean - nilMean) / nilMean
    let deltaPercent = deltaFraction * 100.0

    print("""
      [TelemetryOverheadTests] Results (\(trialCount) trials, alternating order):
        nil  mean: \(String(format: "%.4f", nilMean * 1000)) ms
        Noop mean: \(String(format: "%.4f", noopMean * 1000)) ms
        delta:     \(String(format: "%+.2f", deltaPercent))%
        threshold: <10%
      """)

    #expect(
      noopMean < nilMean * 1.10,
      """
      NoopTuberiaTelemetryReporter overhead exceeded 10%: \
      nilMean=\(String(format: "%.4f", nilMean * 1000))ms, \
      noopMean=\(String(format: "%.4f", noopMean * 1000))ms, \
      delta=\(String(format: "%+.2f", deltaPercent))%
      """
    )
  }
}

// MARK: - ContinuousClock.Duration helper

private extension ContinuousClock.Duration {
  /// Seconds as a `Double`.
  var timeInterval: Double {
    let components = self.components
    return Double(components.seconds) + Double(components.attoseconds) * 1e-18
  }
}
#endif // TUBERIA_TELEMETRY_OVERHEAD_ENABLED
