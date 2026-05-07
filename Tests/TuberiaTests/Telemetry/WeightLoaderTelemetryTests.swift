import Foundation
import Testing

@testable import Tuberia

// MARK: - WeightLoader Telemetry Tests
//
// Design note (Sortie 9 decision):
//   WeightLoader.load and WeightLoader.loadFromPath accept a `telemetry:` parameter,
//   but the current implementation does NOT emit any events itself — the parameter is
//   intentionally plumbing-only (`_ = telemetry`). Telemetry emission for weight loading
//   (e.g. `segmentWeightsLoaded`) is the responsibility of the surrounding pipeline
//   (DiffusionPipeline.loadModels), NOT WeightLoader. This design allows WeightLoader to
//   remain a stateless utility with no actor-hop cost on the hot path, while reserving
//   the parameter for a future revision that may add loader-internal events (sharded-file
//   progress, integrity-check timings). These tests verify that contract.
//
// Synthetic fixture — pure Foundation, no GPU:
//   Builds a minimal valid safetensors binary directly from Foundation byte primitives.
//   No MLX tensor operations, no GPU allocation, no network.
//
//   safetensors format (https://huggingface.co/docs/safetensors/index#format):
//     - 8 bytes: little-endian UInt64 — byte length N of the JSON header
//     - N bytes: UTF-8 JSON metadata dictionary, e.g.:
//         {"test_weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
//     - tensor data bytes (one float32 = 4 bytes at IEEE 754 1.0f = 0x3F800000)
//   Total for a single 1-element float32: 8 + len(json) + 4 bytes.

// MARK: - Fixture helpers

/// Creates a unique temporary directory. The caller must clean up (use a defer block).
private func makeTempDir(label: String) throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("WeightLoaderTelemetry-\(label)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

/// Removes a URL silently (never throws).
private func silentRemove(_ url: URL) {
    try? FileManager.default.removeItem(at: url)
}

/// Writes a minimal valid safetensors binary to `dir/weights.safetensors` using
/// pure Foundation (no MLX, no GPU). The tensor is a single float32 value of 1.0.
///
/// Binary layout:
///   [8 bytes LE UInt64 = header JSON byte count]
///   [header JSON bytes]
///   [4 bytes: float32 1.0f as little-endian IEEE 754 = 0x3F 0x80 0x00 0x00]
///
/// - Returns: URL of the created file.
@discardableResult
private func writeSyntheticSafetensors(in dir: URL, key: String = "test_weight") throws -> URL {
    let fileURL = dir.appendingPathComponent("weights.safetensors")

    // Build the JSON header for a single 1-element float32 tensor.
    let headerJSON = "{\"\(key)\": {\"dtype\": \"F32\", \"shape\": [1], \"data_offsets\": [0, 4]}}"
    guard let headerBytes = headerJSON.data(using: .utf8) else {
        throw CocoaError(.fileWriteUnknown)
    }

    // Encode header byte-count as little-endian 8-byte UInt64.
    var headerLen = UInt64(headerBytes.count).littleEndian
    var fileData = Data(bytes: &headerLen, count: 8)
    fileData.append(headerBytes)

    // Single float32 1.0f in little-endian IEEE 754: 0x3F800000
    let floatValue: Float = 1.0
    var floatBytes = floatValue.bitPattern.littleEndian
    fileData.append(Data(bytes: &floatBytes, count: 4))

    try fileData.write(to: fileURL)
    return fileURL
}

// MARK: - Tests

@Suite("WeightLoader Telemetry Tests", .serialized)
struct WeightLoaderTelemetryTests {

    // MARK: Nil telemetry path

    /// Calling `loadFromPath` with the default nil telemetry succeeds on the synthetic fixture.
    /// Verifies that the telemetry-plumbing addition (Sortie 9) did not break the zero-event
    /// (no reporter) path.
    @Test("loadFromPath with nil telemetry succeeds on synthetic safetensors")
    func weightLoaderWithNilTelemetrySucceeds() async throws {
        let tempDir = try makeTempDir(label: "nil-telemetry")
        defer { silentRemove(tempDir) }

        try writeSyntheticSafetensors(in: tempDir)

        // telemetry: defaults to nil — omitted here to test the default path.
        let params = try await WeightLoader.loadFromPath(
            tempDir.path,
            keyMapping: { key in key }
        )

        #expect(!params.parameters.isEmpty, "Expected at least one parameter from synthetic safetensors")
    }

    // MARK: Mock reporter emits zero events

    /// Passing a MockTuberiaTelemetryReporter to `loadFromPath` records ZERO events.
    ///
    /// Design note: WeightLoader emits no events itself by design (see file header). The
    /// `telemetry:` parameter is reserved plumbing. Asserting event count == 0 is the
    /// correct assertion when calling WeightLoader directly — without going through
    /// DiffusionPipeline, `segmentWeightsLoaded` (which is a pipeline-side emit) never fires.
    @Test("loadFromPath with MockTuberiaTelemetryReporter emits zero events (WeightLoader is plumbing-only)")
    func weightLoaderWithMockReporterEmitsZeroEvents() async throws {
        let tempDir = try makeTempDir(label: "mock-telemetry")
        defer { silentRemove(tempDir) }

        try writeSyntheticSafetensors(in: tempDir)

        let mock = MockTuberiaTelemetryReporter()

        // Pass the mock reporter via the telemetry: parameter (Sortie 9 API).
        // WeightLoader stores the reference as `_ = telemetry` — the reporter is
        // accepted but never called. This is the documented design: WeightLoader is
        // stateless; all telemetry emission happens at the pipeline boundary.
        let params = try await WeightLoader.loadFromPath(
            tempDir.path,
            keyMapping: { key in key },
            telemetry: mock
        )

        #expect(
            !params.parameters.isEmpty,
            "Fixture must have loaded parameters so we know WeightLoader actually ran"
        )

        let events = await mock.events
        // ASSERT: zero events. WeightLoader does not call reporter.capture(_:) by design.
        #expect(
            events.isEmpty,
            "WeightLoader.loadFromPath must emit 0 telemetry events — emits are pipeline-side, not loader-side"
        )
    }

    // MARK: Explicit telemetry: nil produces zero events (symmetry check)

    /// Explicitly passing `telemetry: nil` produces zero events — same result as the default.
    @Test("loadFromPath with explicit telemetry: nil produces zero events")
    func weightLoaderWithExplicitNilTelemetrySucceeds() async throws {
        let tempDir = try makeTempDir(label: "explicit-nil")
        defer { silentRemove(tempDir) }

        try writeSyntheticSafetensors(in: tempDir)

        // Explicitly pass nil to exercise the parameter-present-but-nil branch.
        let params = try await WeightLoader.loadFromPath(
            tempDir.path,
            keyMapping: { key in key },
            telemetry: nil
        )

        #expect(!params.parameters.isEmpty)
    }
}
