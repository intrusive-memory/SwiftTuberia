import Foundation
@preconcurrency import MLX
import MLXRandom
import Testing

@testable import Tuberia

// MARK: - §7 Row 6 — LoRA telemetry
//
// Tests:
//   - Run generate() with a LoRA config pointing to a temporary safetensors file.
//   - Assert loraLoadStart → loraLoadComplete → loraApplied → loraUnapplied
//     fire in that order.

// MARK: - Helpers

extension TuberiaTelemetryLoRATests {

  /// Create a temporary directory with a minimal valid safetensors file containing
  /// a LoRA adapter pair for `layer.0.weight`.
  ///
  /// Uses `MLX.save(arrays:url:)` which writes the native safetensors format.
  /// The file is cleaned up by the test itself via the returned `URL` to the
  /// temporary directory.
  ///
  /// Returns the path to the temporary directory (passed to `LoRAConfig(localPath:)`).
  static func makeTemporaryLoRAFile() throws -> (dirURL: URL, cleanup: () -> Void) {
    let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent(
      "TuberiaTelemetryLoRATests-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

    let safetensorsURL = tmpDir.appendingPathComponent("adapter.safetensors")

    // Minimal LoRA adapter: rank-1 decomposition of a 2×2 weight
    // layer.0.weight.lora_A: [1, 2]
    // layer.0.weight.lora_B: [2, 1]
    let loraA = MLXArray([Float32(0.01), 0.02]).reshaped([1, 2])
    let loraB = MLXArray([Float32(0.01), 0.01]).reshaped([2, 1])
    eval(loraA, loraB)

    try MLX.save(
      arrays: [
        "layer.0.weight.lora_A": loraA,
        "layer.0.weight.lora_B": loraB,
      ],
      url: safetensorsURL
    )

    let cleanup: () -> Void = {
      try? FileManager.default.removeItem(at: tmpDir)
    }
    return (tmpDir, cleanup)
  }
}

// MARK: - Tests

@Suite("TuberiaTelemetryLoRATests — §7 row 6", .serialized)
struct TuberiaTelemetryLoRATests {

  private func makePipeline(rec: RecordingTelemetryReporter) async throws -> DenoisePipeline {
    let pipeline = try DenoisePipeline(recipe: DenoiseNoCFGRecipe(), telemetry: rec)
    await pipeline.setMemoryGate { _ in /* no-op */ }
    return pipeline
  }

  @Test(
    "LoRA load/apply/unapply events fire in order, loraLoadStart carries the expected localPath and scale, and loraUnapplied fires before generate() returns"
  )
  func loraTelemetry() async throws {
    MLXRandom.seed(42)
    TestEnvironment.ensureAcervoAppGroup()

    let (tmpDir, cleanup) = try TuberiaTelemetryLoRATests.makeTemporaryLoRAFile()
    defer { cleanup() }

    let rec = RecordingTelemetryReporter()
    let pipeline = try await makePipeline(rec: rec)

    let loraConfig = LoRAConfig(
      localPath: tmpDir.path,
      scale: 0.5
    )
    let request = DiffusionGenerationRequest(
      prompt: "lora telemetry test",
      width: 64, height: 64,
      steps: 2,
      guidanceScale: 1.0,
      seed: 0,
      loRA: loraConfig
    )
    _ = try await pipeline.generate(request: request, progress: { _ in })

    let events = await rec.events

    // (1) All four LoRA events emitted, in order: loadStart → loadComplete → applied → unapplied
    func index(of kind: String) -> Int? {
      for (i, e) in events.enumerated() {
        switch e {
        case .loraLoadStart where kind == "loraLoadStart": return i
        case .loraLoadComplete where kind == "loraLoadComplete": return i
        case .loraApplied where kind == "loraApplied": return i
        case .loraUnapplied where kind == "loraUnapplied": return i
        default: continue
        }
      }
      return nil
    }

    let loadStartIdx = index(of: "loraLoadStart")
    let loadCompleteIdx = index(of: "loraLoadComplete")
    let appliedIdx = index(of: "loraApplied")
    let unappliedIdx = index(of: "loraUnapplied")

    #expect(loadStartIdx != nil, "loraLoadStart was not emitted")
    #expect(loadCompleteIdx != nil, "loraLoadComplete was not emitted")
    #expect(appliedIdx != nil, "loraApplied was not emitted")
    #expect(
      unappliedIdx != nil,
      "loraUnapplied was not emitted (must fire before generate() returns in the success path)")

    if let s = loadStartIdx, let c = loadCompleteIdx {
      #expect(s < c, "loraLoadStart (\(s)) must precede loraLoadComplete (\(c))")
    }
    if let c = loadCompleteIdx, let a = appliedIdx {
      #expect(c < a, "loraLoadComplete (\(c)) must precede loraApplied (\(a))")
    }
    if let a = appliedIdx, let u = unappliedIdx {
      #expect(a < u, "loraApplied (\(a)) must precede loraUnapplied (\(u))")
    }

    // (2) loraLoadStart carries the expected localPath and scale
    let startEvent = events.compactMap {
      e -> (componentID: String?, localPath: String?, scale: Double, activationKeyword: String?)? in
      if case .loraLoadStart(let cid, let lp, let sc, let kw) = e {
        return (cid, lp, sc, kw)
      } else {
        return nil
      }
    }.first

    if let start = startEvent {
      #expect(
        start.localPath == tmpDir.path,
        "Expected localPath '\(tmpDir.path)', got '\(String(describing: start.localPath))'"
      )
      #expect(abs(start.scale - 0.5) < 0.001, "Expected scale 0.5, got \(start.scale)")
    }
  }
}
