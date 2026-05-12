import Foundation
@preconcurrency import MLX
import MLXRandom
import Testing

@testable import Tuberia

// MARK: - §7 Row 3 — CFG dtype-cast telemetry
//
// Drives one CFG run (guidanceScale > 1.0) and asserts every property of the
// emitted `cfgDtypeCast` events: per-step emission count, fromDtype, toDtype,
// and guidedPredictionStat.dtype. All assertions land against the same
// captured event stream so a single pipeline run covers the whole row.

@Suite("TuberiaTelemetryCFGCastTests — §7 row 3", .serialized)
struct TuberiaTelemetryCFGCastTests {

  private func makePipeline(rec: RecordingTelemetryReporter) async throws -> DenoisePipeline {
    let pipeline = try DenoisePipeline(recipe: DenoiseCFGRecipe(), telemetry: rec)
    await pipeline.setMemoryGate { _ in /* no-op */ }
    return pipeline
  }

  @Test(
    "cfgDtypeCast fires per step with correct fromDtype, toDtype, and guidedPredictionStat.dtype")
  func cfgDtypeCastTelemetry() async throws {
    MLXRandom.seed(42)
    let rec = RecordingTelemetryReporter()
    let pipeline = try await makePipeline(rec: rec)

    // guidanceScale > 1.0 activates the CFG branch
    let request = DiffusionGenerationRequest(
      prompt: "CFG cast test",
      width: 64, height: 64,
      steps: 4,
      guidanceScale: 7.5,
      seed: 0
    )
    _ = try await pipeline.generate(request: request, progress: { _ in })

    let events = await rec.events
    let castEvents = events.compactMap {
      e -> (stepIndex: Int, fromDtype: String, toDtype: String, stat: TuberiaTensorStat)? in
      if case .cfgDtypeCast(let idx, let from, let to, let stat) = e {
        return (idx, from, to, stat)
      } else {
        return nil
      }
    }

    // (1) One cfgDtypeCast per step
    #expect(
      castEvents.count == 4,
      "Expected 4 cfgDtypeCast events (one per step), got \(castEvents.count)"
    )

    // (2) fromDtype == "float16" (backbone output dtype) on every cast
    // (3) toDtype == "float32" on every cast
    // (4) guidedPredictionStat.dtype == "float32" on every cast
    for cast in castEvents {
      #expect(
        cast.fromDtype == "float16",
        "Expected fromDtype 'float16' at step \(cast.stepIndex), got '\(cast.fromDtype)'"
      )
      #expect(
        cast.toDtype == "float32",
        "Expected toDtype 'float32' at step \(cast.stepIndex), got '\(cast.toDtype)'"
      )
      #expect(
        cast.stat.dtype == "float32",
        "Expected guidedPredictionStat.dtype 'float32' at step \(cast.stepIndex), got '\(cast.stat.dtype)'"
      )
    }
  }
}
