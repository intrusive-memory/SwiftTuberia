import CoreGraphics
import Foundation
@preconcurrency import MLX
import os
import SwiftAcervo
import Testing

@testable import Tuberia

// MARK: - In-target stub components (TuberiaTests-only, no dependency on TuberiaGPUTests)

private final class TelemetryStubEncoder: TextEncoder, @unchecked Sendable {
  struct Config: Sendable {
    let dim: Int
    let seqLen: Int
  }
  typealias Configuration = Config
  private(set) var isLoaded = false
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

private final class TelemetryStubBackbone: Backbone, @unchecked Sendable {
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
  var estimatedMemoryBytes: Int { 2_000 }
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

private final class TelemetryStubDecoder: Decoder, @unchecked Sendable {
  struct Config: Sendable {}
  typealias Configuration = Config
  private(set) var isLoaded = false
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

private final class TelemetryStubScheduler: Scheduler, @unchecked Sendable {
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

private final class TelemetryStubRenderer: Renderer, @unchecked Sendable {
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
      throw PipelineError.renderingFailed(reason: "TelemetryStubRenderer: CGImage failed")
    }
    return .image(img)
  }
}

/// Minimal CPU-only stub recipe for telemetry tests.
/// Empty `allComponentIds` → no WeightLoader calls → `loadModels` completes synchronously
/// without real file I/O, exercising all telemetry emit sites except
/// `segmentWeightsLoaded` / `segmentApplied` (those require non-nil component IDs).
private struct DiffusionTelemetryStubRecipe: PipelineRecipe, Sendable {
  typealias Encoder = TelemetryStubEncoder
  typealias Sched = TelemetryStubScheduler
  typealias Back = TelemetryStubBackbone
  typealias Dec = TelemetryStubDecoder
  typealias Rend = TelemetryStubRenderer

  let encoderConfig = TelemetryStubEncoder.Config(dim: 64, seqLen: 8)
  let schedulerConfig = TelemetryStubScheduler.Config()
  let backboneConfig = TelemetryStubBackbone.Config(dim: 64, seqLen: 8)
  let decoderConfig = TelemetryStubDecoder.Config()
  let rendererConfig: Void = ()

  let supportsImageToImage = false
  let unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .emptyPrompt
  /// Empty component IDs: no WeightLoader calls, so `loadModels` always succeeds.
  let allComponentIds: [String] = []

  func quantizationFor(_ role: PipelineRole) -> QuantizationConfig { .asStored }
  func validate() throws {}
}

// MARK: - Thread-safe counter for @Sendable progress closures

/// Lock-protected counter for incrementing from @Sendable (synchronous) closures.
private final class ProgressCounter: @unchecked Sendable {
  private let lock = OSAllocatedUnfairLock(initialState: 0)
  func increment() { lock.withLock { $0 += 1 } }
  var value: Int { lock.withLock { $0 } }
}

// MARK: - Case-discrimination helpers
// Use if-case patterns rather than positional array access; conditional emits
// (e.g. segmentDownloadProgress, rate-limited) may interleave unpredictably.

private func isLoadStart(_ event: TuberiaTelemetryEvent) -> Bool {
  if case .loadStart = event { return true }
  return false
}

private func isMemoryGate(_ event: TuberiaTelemetryEvent) -> Bool {
  if case .memoryGate = event { return true }
  return false
}

private func isSegmentLoadStart(_ event: TuberiaTelemetryEvent) -> Bool {
  if case .segmentLoadStart = event { return true }
  return false
}

private func isSegmentWeightsLoaded(_ event: TuberiaTelemetryEvent) -> Bool {
  if case .segmentWeightsLoaded = event { return true }
  return false
}

private func isSegmentApplied(_ event: TuberiaTelemetryEvent) -> Bool {
  if case .segmentApplied = event { return true }
  return false
}

private func isLoadComplete(_ event: TuberiaTelemetryEvent) -> Bool {
  if case .loadComplete = event { return true }
  return false
}

private func isUnloadSegment(_ event: TuberiaTelemetryEvent) -> Bool {
  if case .unloadSegment = event { return true }
  return false
}

private func isErrorThrown(_ event: TuberiaTelemetryEvent) -> Bool {
  if case .errorThrown = event { return true }
  return false
}

// MARK: - Test Suite

@Suite("DiffusionPipeline Telemetry Tests", .serialized)
struct DiffusionPipelineTelemetryTests {

  init() { TestEnvironment.ensureAcervoAppGroup() }

  // MARK: - Pipeline type alias

  private typealias TestPipeline = DiffusionPipeline<
    TelemetryStubEncoder,
    TelemetryStubScheduler,
    TelemetryStubBackbone,
    TelemetryStubDecoder,
    TelemetryStubRenderer
  >

  private func makePipeline() throws -> TestPipeline {
    try TestPipeline(recipe: DiffusionTelemetryStubRecipe())
  }

  // MARK: - Test 1: Happy-path load event ordering
  //
  // Asserts the strict lifecycle sequence:
  //   loadStart → memoryGate(passed:true) → segmentLoadStart(×3) → loadComplete
  //
  // With empty allComponentIds, WeightLoader is never invoked.
  // segmentWeightsLoaded / segmentApplied are NOT emitted (nil component IDs)
  // and must NOT appear in the event stream.

  @Test("loadModels emits loadStart before memoryGate before segmentLoadStart before loadComplete")
  func happyPathLoadOrdering() async throws {
    let pipeline = try makePipeline()
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)
    // Stub memory gate to pass unconditionally (avoids hardware query side-effects).
    await pipeline.setMemoryGate { _ in }

    try await pipeline.loadModels { _, _ in }

    let events = await mock.events

    // -- Primary ordering assertions --

    // Locate event indices by case discriminator (not position) to tolerate any
    // conditionally-emitted events that may interleave.
    let loadStartIdx = events.firstIndex(where: isLoadStart)
    let memoryGateIdx = events.firstIndex(where: isMemoryGate)
    let firstSegmentLoadStartIdx = events.firstIndex(where: isSegmentLoadStart)
    let loadCompleteIdx = events.firstIndex(where: isLoadComplete)

    let loadStartIndex = try #require(loadStartIdx, "loadStart must be emitted")
    let memoryGateIndex = try #require(memoryGateIdx, "memoryGate must be emitted")
    let segmentLoadStartIndex = try #require(
      firstSegmentLoadStartIdx, "segmentLoadStart must be emitted")
    let loadCompleteIndex = try #require(loadCompleteIdx, "loadComplete must be emitted")

    #expect(loadStartIndex < memoryGateIndex, "loadStart must precede memoryGate")
    #expect(memoryGateIndex < segmentLoadStartIndex, "memoryGate must precede segmentLoadStart")
    #expect(segmentLoadStartIndex < loadCompleteIndex, "segmentLoadStart must precede loadComplete")
  }

  // MARK: - Test 2: memoryGate passed == true on happy path

  @Test("loadModels emits memoryGate with passed == true on happy path")
  func happyPathMemoryGatePassed() async throws {
    let pipeline = try makePipeline()
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)
    await pipeline.setMemoryGate { _ in }

    try await pipeline.loadModels { _, _ in }

    let events = await mock.events
    let memoryGateEvents = events.filter(isMemoryGate)

    // Exactly one memoryGate event on the success path.
    #expect(memoryGateEvents.count == 1, "Exactly one memoryGate event should be emitted")

    guard let gateEvent = memoryGateEvents.first else { return }
    if case .memoryGate(_, _, let passed) = gateEvent {
      #expect(passed == true, "memoryGate must report passed == true on the happy path")
    } else {
      Issue.record("Expected .memoryGate event; got \(gateEvent)")
    }
  }

  // MARK: - Test 3: Failure path — memoryGate throws

  @Test("loadModels emits memoryGate(passed:false) then errorThrown when gate throws")
  func failurePathMemoryGateThrows() async throws {
    let pipeline = try makePipeline()
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)

    let simulatedAvailable: UInt64 = 1
    // Inject a gate that always throws PipelineError.insufficientMemory.
    await pipeline.setMemoryGate { requiredBytes in
      throw PipelineError.insufficientMemory(
        required: requiredBytes,
        available: simulatedAvailable,
        component: "pipeline"
      )
    }

    var didCatchError = false
    do {
      try await pipeline.loadModels { _, _ in }
      Issue.record("Expected PipelineError.insufficientMemory but loadModels succeeded")
    } catch let error as PipelineError {
      if case .insufficientMemory = error {
        didCatchError = true
      } else {
        Issue.record("Expected .insufficientMemory, got \(error)")
      }
    } catch {
      Issue.record("Expected PipelineError, got \(error)")
    }

    #expect(didCatchError, "Caller must observe the rethrown PipelineError.insufficientMemory")

    let events = await mock.events

    // memoryGate(passed:false) must be present.
    let gateEvents = events.filter(isMemoryGate)
    #expect(gateEvents.count >= 1, "memoryGate must be emitted even on failure")
    if let gateEvent = gateEvents.first, case .memoryGate(_, _, let passed) = gateEvent {
      #expect(passed == false, "Failed gate must emit memoryGate(passed: false)")
    }

    // errorThrown with phase "loadModels.memoryGate" must be present.
    let errorEvents = events.filter(isErrorThrown)
    #expect(errorEvents.count >= 1, "errorThrown must be emitted after gate failure")
    let memoryGateErrorEvent = errorEvents.first { event in
      if case .errorThrown(let phase, _) = event { return phase == "loadModels.memoryGate" }
      return false
    }
    #expect(memoryGateErrorEvent != nil, "errorThrown(phase: \"loadModels.memoryGate\") must be emitted")

    // memoryGate(passed:false) must precede errorThrown in the event stream.
    let gateFailedIdx = events.firstIndex(where: {
      if case .memoryGate(_, _, let passed) = $0 { return !passed }
      return false
    })
    let errorThrownIdx = events.firstIndex(where: {
      if case .errorThrown(let phase, _) = $0 { return phase == "loadModels.memoryGate" }
      return false
    })
    if let gi = gateFailedIdx, let ei = errorThrownIdx {
      #expect(gi < ei, "memoryGate(passed:false) must precede errorThrown in event order")
    }
  }

  // MARK: - Test 4: segmentLoadStart fires once per weighted segment

  @Test("loadModels emits segmentLoadStart exactly once per weighted segment in canonical order")
  func segmentLoadStartFiresOncePerSegment() async throws {
    let pipeline = try makePipeline()
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)
    await pipeline.setMemoryGate { _ in }

    try await pipeline.loadModels { _, _ in }

    let events = await mock.events
    let loadStartEvents = events.filter(isSegmentLoadStart)

    // The pipeline always has three weighted segments: encoder, backbone, decoder.
    #expect(
      loadStartEvents.count == 3,
      "segmentLoadStart must fire exactly 3 times (encoder, backbone, decoder)")

    // Verify canonical role order: encoder → backbone → decoder.
    let roles: [PipelineRole] = loadStartEvents.compactMap { event in
      if case .segmentLoadStart(let role, _, _, _) = event { return role }
      return nil
    }
    #expect(roles == [.encoder, .backbone, .decoder], "segmentLoadStart roles must appear in canonical order: encoder, backbone, decoder")
  }

  // MARK: - Test 5: segmentWeightsLoaded / segmentApplied absent when allComponentIds is empty

  @Test("segmentWeightsLoaded and segmentApplied are not emitted when allComponentIds is empty")
  func segmentWeightsLoadedAndAppliedAbsentWithNilIds() async throws {
    // With no component IDs, WeightLoader is never invoked.
    // The pipeline loops over the 3 segments but skips the load block for each.
    // Therefore segmentWeightsLoaded and segmentApplied must not appear.
    let pipeline = try makePipeline()
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)
    await pipeline.setMemoryGate { _ in }

    try await pipeline.loadModels { _, _ in }

    let events = await mock.events
    let weightsLoadedEvents = events.filter(isSegmentWeightsLoaded)
    let appliedEvents = events.filter(isSegmentApplied)

    #expect(
      weightsLoadedEvents.isEmpty,
      "segmentWeightsLoaded must not fire when allComponentIds is empty (no WeightLoader calls)")
    #expect(
      appliedEvents.isEmpty,
      "segmentApplied must not fire when allComponentIds is empty (no WeightLoader calls)")
  }

  // MARK: - Test 6: loadComplete.totalEstimatedBytes == sum of segmentApplied.estimatedMemoryBytes

  @Test("loadComplete.totalEstimatedBytes equals sum of segmentApplied.estimatedMemoryBytes")
  func loadCompleteTotalBytesEqualsSegmentAppliedSum() async throws {
    // With empty component IDs, no segmentApplied events fire and totalEstimatedBytes == 0.
    // Both sides of the invariant are 0 — this confirms the accumulation logic holds.
    let pipeline = try makePipeline()
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)
    await pipeline.setMemoryGate { _ in }

    try await pipeline.loadModels { _, _ in }

    let events = await mock.events

    let totalFromApplied: UInt64 = events.reduce(0) { acc, event in
      if case .segmentApplied(_, _, let bytes) = event { return acc + bytes }
      return acc
    }

    guard let completeEvent = events.first(where: isLoadComplete) else {
      Issue.record("loadComplete event not found")
      return
    }

    if case .loadComplete(_, let totalEstimatedBytes) = completeEvent {
      #expect(
        totalEstimatedBytes == totalFromApplied,
        "loadComplete.totalEstimatedBytes (\(totalEstimatedBytes)) must equal sum of segmentApplied.estimatedMemoryBytes (\(totalFromApplied))")
    } else {
      Issue.record("Expected .loadComplete event")
    }
  }

  // MARK: - Test 7: unloadModels emits one unloadSegment per loaded segment

  @Test("unloadModels emits one unloadSegment per segment (encoder, backbone, decoder)")
  func unloadModelsEmitsOneUnloadSegmentPerSegment() async throws {
    let pipeline = try makePipeline()
    let mock = MockTuberiaTelemetryReporter()
    await pipeline.setTelemetry(mock)
    await pipeline.setMemoryGate { _ in }

    // Load first so unload has something to unload.
    try await pipeline.loadModels { _, _ in }
    _ = await mock.drain()  // Clear load-phase events; focus on unload events only.

    await pipeline.unloadModels()

    let events = await mock.events
    // Filter only unloadSegment events with non-nil roles (Sortie 9 will test role:nil separately).
    let unloadEvents = events.filter { event in
      if case .unloadSegment(let role, _, _, _) = event { return role != nil }
      return false
    }

    #expect(
      unloadEvents.count == 3,
      "unloadModels must emit exactly 3 unloadSegment events (encoder, backbone, decoder); got \(unloadEvents.count)")

    // Verify exact role order: encoder → backbone → decoder.
    let roles: [PipelineRole] = unloadEvents.compactMap { event in
      if case .unloadSegment(let role, _, _, _) = event { return role }
      return nil
    }
    #expect(
      roles == [.encoder, .backbone, .decoder],
      "unloadSegment events must appear in canonical order: encoder, backbone, decoder")

    // Metal byte fields must be nil (instrumented in Sortie 9, not here).
    for event in unloadEvents {
      if case .unloadSegment(_, _, let before, let after) = event {
        #expect(before == nil, "metalAllocatedBytesBefore must be nil for per-segment unloadSegment events (Metal instrumented in Sortie 9)")
        #expect(after == nil, "metalAllocatedBytesAfter must be nil for per-segment unloadSegment events (Metal instrumented in Sortie 9)")
      }
    }
  }

  // MARK: - Test 8: Default nil reporter — no events observable

  @Test("No telemetry reporter: pipeline runs without crashes and emits no events")
  func defaultNilReporterEmitsNoEvents() async throws {
    // Create pipeline without attaching any telemetry reporter.
    // The nil default means emit(_:) is a no-op; the pipeline must still complete normally.
    let pipeline = try makePipeline()
    await pipeline.setMemoryGate { _ in }
    // (no setTelemetry call — default is nil)

    // Attach a separate control reporter to confirm the pipeline ran (via progress callback).
    // Use a lock-isolated counter to satisfy Swift 6 concurrency requirements (@Sendable closure).
    let progressCounter = ProgressCounter()
    try await pipeline.loadModels { _, _ in
      progressCounter.increment()
    }
    let progressCallCount = progressCounter.value

    #expect(progressCallCount > 0, "Progress callback must be called even without a telemetry reporter")

    // Now attach a reporter and run a fresh pipeline to verify telemetry does appear
    // when a reporter IS wired in.  This validates the "no-op when nil" contract by
    // contrast: the same pipeline type emits events only when a reporter is set.
    let pipeline2 = try makePipeline()
    let mock = MockTuberiaTelemetryReporter()
    await pipeline2.setTelemetry(mock)
    await pipeline2.setMemoryGate { _ in }
    try await pipeline2.loadModels { _, _ in }

    let eventsWithReporter = await mock.events
    #expect(!eventsWithReporter.isEmpty, "Events must be recorded when a reporter is attached")
  }
}
