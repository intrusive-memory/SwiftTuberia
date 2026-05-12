import Foundation
@preconcurrency import MLX
import MLXRandom
import Testing

@testable import Tuberia

// MARK: - §7 Row 4 — Numerical anomaly telemetry
//
// Tests:
//   (a) `denoiseStepComplete` for step 2 carries `predictionStat.hasNaN == true`.
//   (b) `numericalAnomaly(phase: "backbone_forward_complete_no_cfg", kind: .nan,
//       stepIndex: 2, ...)` fires within the same step boundary.
//
// NOTE: The production code emits a more specific phase string than the requirements
// doc's abbreviated `"backbone_forward_complete"`. The actual strings depend on
// which branch runs:
//   - No-CFG: "backbone_forward_complete_no_cfg"
//   - CFG conditional: "backbone_forward_complete_cfg_conditional"
//   - CFG unconditional: "backbone_forward_complete_cfg_unconditional"
//
// This test uses the no-CFG path (guidanceScale == 1.0) so the phase is
// "backbone_forward_complete_no_cfg".

// MARK: - NaN-Injecting Backbone

/// Subclass of `DenoiseBackbone` that injects NaN at the configured step index.
final class NaNInjectingBackbone: DenoiseBackbone, @unchecked Sendable {
  private let nanAtStep: Int
  private var stepCount: Int = 0

  init(config: Config, nanAtStep: Int) throws {
    self.nanAtStep = nanAtStep
    try super.init(configuration: config)
  }

  required init(configuration: Config) throws {
    self.nanAtStep = 2
    try super.init(configuration: configuration)
  }

  override func forward(_ input: BackboneInput) throws -> MLXArray {
    let result = try super.forward(input)
    let currentStep = stepCount
    stepCount += 1
    if currentStep == nanAtStep {
      // Build a tensor: mostly zeros with NaN at position 0
      // to trigger the hasNaN path in TuberiaTensorStat.sample
      let flat = result.flattened()
      var values = flat.asArray(Float.self)
      if !values.isEmpty {
        values[0] = Float.nan
      }
      let nanArray = MLXArray(values).reshaped(result.shape).asType(.float16)
      return nanArray
    }
    return result
  }
}

// MARK: - Recipe

private struct AnomalyRecipe: PipelineRecipe, Sendable {
  typealias Encoder = DenoiseEncoder
  typealias Sched = DenoiseScheduler
  typealias Back = NaNInjectingBackbone
  typealias Dec = DenoiseDecoder
  typealias Rend = DenoiseRenderer

  let encoderConfig = DenoiseEncoder.Config(dim: 64, seqLen: 8)
  let schedulerConfig = DenoiseScheduler.Config(predType: "epsilon")
  let backboneConfig = NaNInjectingBackbone.Config(condDim: 64, seqLen: 8, latentChannels: 4)
  let decoderConfig = DenoiseDecoder.Config(inputChannels: 4)
  let rendererConfig: Void = ()

  let supportsImageToImage = false
  let unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .emptyPrompt
  let allComponentIds: [String] = []
  func quantizationFor(_ role: PipelineRole) -> QuantizationConfig { .asStored }
  func validate() throws {}
}

private typealias AnomalyPipeline = DiffusionPipeline<
  DenoiseEncoder,
  DenoiseScheduler,
  NaNInjectingBackbone,
  DenoiseDecoder,
  DenoiseRenderer
>

// MARK: - Tests

@Suite("TuberiaTelemetryAnomalyTests — §7 row 4", .serialized)
struct TuberiaTelemetryAnomalyTests {

  private func makePipeline(rec: RecordingTelemetryReporter) async throws -> AnomalyPipeline {
    let pipeline = try AnomalyPipeline(recipe: AnomalyRecipe(), telemetry: rec)
    await pipeline.setMemoryGate { _ in /* no-op */ }
    return pipeline
  }

  @Test(
    "NaN injection at step 2 surfaces in predictionStat, fires numericalAnomaly(.nan), and the anomaly falls within the step 2 boundary"
  )
  func nanInjectionTelemetry() async throws {
    MLXRandom.seed(42)
    let rec = RecordingTelemetryReporter()
    let pipeline = try await makePipeline(rec: rec)

    let request = DiffusionGenerationRequest(
      prompt: "nan injection test",
      width: 64, height: 64,
      steps: 4,
      guidanceScale: 1.0,  // no-CFG path — single backbone forward per step
      seed: 0
    )
    // NaN in the prediction propagates to latents; pipeline may or may not throw.
    // Either way, assertions are on the recorder's captured events.
    _ = try? await pipeline.generate(request: request, progress: { _ in })

    let events = await rec.events

    // (1) denoiseStepComplete for stepIndex == 2 carries predictionStat.hasNaN == true
    let step2Complete = events.compactMap { e -> TuberiaTelemetryEvent? in
      if case .denoiseStepComplete(let idx, _, _, _, _, _, _) = e, idx == 2 {
        return e
      } else {
        return nil
      }
    }.first

    #expect(step2Complete != nil, "Expected denoiseStepComplete(stepIndex: 2) to be emitted")
    if case .denoiseStepComplete(_, _, _, _, _, let predictionStat, _) = step2Complete {
      #expect(
        predictionStat.hasNaN,
        "Expected predictionStat.hasNaN == true for step 2, got false"
      )
    }

    // (2) numericalAnomaly(.nan, stepIndex: 2) fires, with a phase in the
    //     backbone_forward_complete_* family (no-CFG variant: "_no_cfg")
    let nanAnomalies = events.compactMap {
      e -> (phase: String, kind: TuberiaTelemetryEvent.AnomalyKind, stepIndex: Int?)? in
      if case .numericalAnomaly(let phase, let kind, let stepIdx, _) = e {
        return (phase, kind, stepIdx)
      } else {
        return nil
      }
    }.filter { $0.kind == .nan && $0.stepIndex == 2 }

    #expect(
      !nanAnomalies.isEmpty,
      "Expected at least one numericalAnomaly(.nan, stepIndex: 2) event"
    )
    let backbonePhases = nanAnomalies.filter {
      $0.phase.hasPrefix("backbone_forward_complete")
    }
    #expect(
      !backbonePhases.isEmpty,
      "Expected numericalAnomaly(.nan) to have a backbone_forward_complete phase, got: \(nanAnomalies.map { $0.phase })"
    )

    // (3) The anomaly event falls inside the step 2 boundary
    //     (after step 2 start, before step 3 start if step 3 fires)
    var step2StartIdx: Int? = nil
    var step3StartIdx: Int? = nil
    for (i, e) in events.enumerated() {
      if case .denoiseStepStart(let idx, _, _, _, _, _) = e {
        if idx == 2 { step2StartIdx = i }
        if idx == 3 { step3StartIdx = i }
      }
    }
    let nanAnomalyIdx = events.indices.first { i in
      if case .numericalAnomaly(_, let kind, let sIdx, _) = events[i] {
        return kind == .nan && sIdx == 2
      }
      return false
    }

    if let start2 = step2StartIdx, let anomalyIdx = nanAnomalyIdx {
      #expect(
        anomalyIdx > start2,
        "Anomaly at \(anomalyIdx) must come after step2Start at \(start2)"
      )
      if let start3 = step3StartIdx {
        #expect(
          anomalyIdx < start3,
          "Anomaly at \(anomalyIdx) must come before step3Start at \(start3)"
        )
      }
    }
  }
}
