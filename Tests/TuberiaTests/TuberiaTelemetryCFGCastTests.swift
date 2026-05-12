import Foundation
@preconcurrency import MLX
import MLXRandom
import Testing

@testable import Tuberia

// MARK: - §7 Row 3 — CFG dtype-cast telemetry
//
// Tests:
//   - Drive a CFG run (guidanceScale > 1.0).
//   - Assert cfgDtypeCast fires per step.
//   - Assert fromDtype == "float16" (backbone returns float16).
//   - Assert toDtype == "float32".
//   - Assert guidedPredictionStat.dtype == "float32".

@Suite("TuberiaTelemetryCFGCastTests — §7 row 3", .serialized)
struct TuberiaTelemetryCFGCastTests {

  // MARK: - Helpers

  private func makePipeline(rec: RecordingTelemetryReporter) async throws -> DenoisePipeline {
    let pipeline = try DenoisePipeline(recipe: DenoiseCFGRecipe(), telemetry: rec)
    await pipeline.setMemoryGate { _ in /* no-op */ }
    return pipeline
  }

  // MARK: - CFG cast fires per step

  @Test("cfgDtypeCast fires once per step in a 4-step CFG run")
  func cfgDtypeCastFiresPerStep() async throws {
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

    // One cfgDtypeCast per step
    #expect(
      castEvents.count == 4,
      "Expected 4 cfgDtypeCast events (one per step), got \(castEvents.count)"
    )
  }

  @Test("cfgDtypeCast.fromDtype is float16 (backbone output dtype)")
  func cfgDtypeCastFromDtypeIsFloat16() async throws {
    MLXRandom.seed(42)
    let rec = RecordingTelemetryReporter()
    let pipeline = try await makePipeline(rec: rec)

    let request = DiffusionGenerationRequest(
      prompt: "CFG dtype from test",
      width: 64, height: 64,
      steps: 4,
      guidanceScale: 7.5,
      seed: 1
    )
    _ = try await pipeline.generate(request: request, progress: { _ in })

    let events = await rec.events
    let castFromDtypes = events.compactMap { e -> String? in
      if case .cfgDtypeCast(_, let from, _, _) = e { return from } else { return nil }
    }

    #expect(!castFromDtypes.isEmpty)
    for fromDtype in castFromDtypes {
      #expect(
        fromDtype == "float16",
        "Expected fromDtype == 'float16', got '\(fromDtype)'"
      )
    }
  }

  @Test("cfgDtypeCast.toDtype is float32")
  func cfgDtypeCastToDtypeIsFloat32() async throws {
    MLXRandom.seed(42)
    let rec = RecordingTelemetryReporter()
    let pipeline = try await makePipeline(rec: rec)

    let request = DiffusionGenerationRequest(
      prompt: "CFG dtype to test",
      width: 64, height: 64,
      steps: 4,
      guidanceScale: 7.5,
      seed: 2
    )
    _ = try await pipeline.generate(request: request, progress: { _ in })

    let events = await rec.events
    let castToDtypes = events.compactMap { e -> String? in
      if case .cfgDtypeCast(_, _, let to, _) = e { return to } else { return nil }
    }

    #expect(!castToDtypes.isEmpty)
    for toDtype in castToDtypes {
      #expect(toDtype == "float32", "Expected toDtype == 'float32', got '\(toDtype)'")
    }
  }

  @Test("cfgDtypeCast.guidedPredictionStat.dtype is float32")
  func cfgDtypeCastGuidedPredictionStatDtypeIsFloat32() async throws {
    MLXRandom.seed(42)
    let rec = RecordingTelemetryReporter()
    let pipeline = try await makePipeline(rec: rec)

    let request = DiffusionGenerationRequest(
      prompt: "guided stat dtype test",
      width: 64, height: 64,
      steps: 4,
      guidanceScale: 7.5,
      seed: 3
    )
    _ = try await pipeline.generate(request: request, progress: { _ in })

    let events = await rec.events
    let guidedStats = events.compactMap { e -> TuberiaTensorStat? in
      if case .cfgDtypeCast(_, _, _, let stat) = e { return stat } else { return nil }
    }

    #expect(!guidedStats.isEmpty)
    for stat in guidedStats {
      #expect(
        stat.dtype == "float32",
        "Expected guidedPredictionStat.dtype == 'float32', got '\(stat.dtype)'"
      )
    }
  }
}
