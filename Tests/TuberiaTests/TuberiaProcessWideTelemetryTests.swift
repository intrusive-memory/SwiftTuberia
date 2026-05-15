import CoreGraphics
import Foundation
@preconcurrency import MLX
import MLXRandom
import Testing

@testable import Tuberia

// MARK: - Process-wide TuberiaTelemetry seam tests
//
// Tests:
//   1. No instance reporter + process-wide reporter set → emissions reach the process-wide reporter.
//   2. Both set → instance reporter wins (process-wide reporter receives no events).
//   3. Setting process-wide reporter to nil restores "no reporter" state.
//   4. A real DiffusionPipeline-level emission path exercises at least one event via
//      the process-wide reporter alone (pipelineConfigured on init).

// MARK: - Helpers

/// A RecordingTelemetryReporter variant named ProcessWideRecorder for clarity in these tests.
private actor ProcessWideRecorder: TuberiaTelemetryReporter {
  private(set) var events: [TuberiaTelemetryEvent] = []

  func capture(_ event: TuberiaTelemetryEvent) async {
    events.append(event)
  }

  func clear() {
    events = []
  }
}

// MARK: - Tests

@Suite("TuberiaProcessWideTelemetryTests — process-wide seam", .serialized)
struct TuberiaProcessWideTelemetryTests {

  // MARK: - Teardown helper

  /// Always clear the process-wide reporter after each test so state does not bleed between tests.
  private func clearProcessWideReporter() {
    TuberiaTelemetry.setReporter(nil)
  }

  // MARK: - Test 1: Process-wide reporter receives events when no instance reporter is set

  @Test(
    "No instance reporter + process-wide reporter set → emissions reach the process-wide reporter")
  func processWideReporterReceivesEventsWhenNoInstanceReporter() async throws {
    defer { clearProcessWideReporter() }

    let processWide = ProcessWideRecorder()
    TuberiaTelemetry.setReporter(processWide)

    // Construct a pipeline with NO instance reporter (telemetry: nil).
    // pipelineConfigured fires in init when effectiveInitTelemetry is non-nil —
    // which it now is thanks to TuberiaTelemetry.current.
    let pipeline = try DenoisePipeline(recipe: DenoiseNoCFGRecipe(), telemetry: nil)
    _ = pipeline

    // Allow the Task {} inside validateAssembly and the pipelineConfigured Task to run.
    let deadline = Date().addingTimeInterval(2.0)
    var events: [TuberiaTelemetryEvent] = []
    while Date() < deadline {
      events = await processWide.events
      let hasPipelineConfigured = events.contains {
        if case .pipelineConfigured = $0 { return true }
        return false
      }
      if hasPipelineConfigured { break }
      await Task.yield()
    }

    #expect(
      !events.isEmpty,
      "Process-wide reporter should have received at least one event"
    )
    let hasPipelineConfigured = events.contains {
      if case .pipelineConfigured = $0 { return true }
      return false
    }
    #expect(
      hasPipelineConfigured,
      "Expected pipelineConfigured event from init via process-wide reporter. Got: \(events.map { "\($0)" })"
    )
  }

  // MARK: - Test 2: Instance reporter wins over process-wide reporter

  @Test("Both reporters set → instance reporter wins, process-wide reporter receives no events")
  func instanceReporterWinsOverProcessWide() async throws {
    defer { clearProcessWideReporter() }

    let instanceRecorder = RecordingTelemetryReporter()
    let processWideRecorder = ProcessWideRecorder()

    TuberiaTelemetry.setReporter(processWideRecorder)

    // Run a full 2-step generate() with the instance reporter set.
    // The instance reporter should receive all events; the process-wide reporter none.
    MLXRandom.seed(42)

    let pipeline = try DenoisePipeline(
      recipe: DenoiseNoCFGRecipe(),
      telemetry: instanceRecorder
    )
    await pipeline.setMemoryGate { _ in /* no-op */ }

    let request = DiffusionGenerationRequest(
      prompt: "instance wins test",
      width: 64, height: 64,
      steps: 2,
      guidanceScale: 1.0,
      seed: 42
    )
    _ = try await pipeline.generate(request: request, progress: { _ in })

    // Give any trailing Task {} a chance to flush
    for _ in 0..<10 { await Task.yield() }

    let instanceEvents = await instanceRecorder.events
    let processWideEvents = await processWideRecorder.events

    #expect(
      !instanceEvents.isEmpty,
      "Instance reporter must receive events when it is set"
    )
    #expect(
      processWideEvents.isEmpty,
      "Process-wide reporter must receive NO events when instance reporter is set. Got: \(processWideEvents.count) events"
    )
  }

  // MARK: - Test 3: setReporter(nil) restores "no reporter" state

  @Test("Setting process-wide reporter to nil restores 'no reporter' state")
  func settingReporterToNilRestoresNoReporterState() async throws {
    defer { clearProcessWideReporter() }

    let processWideRecorder = ProcessWideRecorder()

    // Install then immediately clear the process-wide reporter.
    TuberiaTelemetry.setReporter(processWideRecorder)
    TuberiaTelemetry.setReporter(nil)

    // Verify the accessor reflects nil.
    let current = TuberiaTelemetry.current
    #expect(
      current == nil,
      "TuberiaTelemetry.current should be nil after setReporter(nil)"
    )

    // Construct a pipeline with no instance reporter — should produce no events
    // in the process-wide recorder since the reporter was cleared.
    let pipeline = try DenoisePipeline(recipe: DenoiseNoCFGRecipe(), telemetry: nil)
    _ = pipeline

    // Drain the cooperative executor briefly.
    for _ in 0..<20 { await Task.yield() }

    let events = await processWideRecorder.events
    #expect(
      events.isEmpty,
      "Process-wide recorder should receive no events after setReporter(nil). Got: \(events.count)"
    )
  }

  // MARK: - Test 4: Real DiffusionPipeline emission path via process-wide reporter

  @Test(
    "Real DiffusionPipeline generate() emits pipeline lifecycle events via process-wide reporter")
  func realPipelineEmissionsViaProcessWideReporter() async throws {
    defer { clearProcessWideReporter() }

    MLXRandom.seed(42)

    let processWideRecorder = ProcessWideRecorder()
    TuberiaTelemetry.setReporter(processWideRecorder)

    // Pipeline with NO instance reporter — all events must come via process-wide.
    let pipeline = try DenoisePipeline(recipe: DenoiseNoCFGRecipe(), telemetry: nil)
    await pipeline.setMemoryGate { _ in /* no-op */ }

    let request = DiffusionGenerationRequest(
      prompt: "process-wide pipeline test",
      width: 64, height: 64,
      steps: 2,
      guidanceScale: 1.0,
      seed: 42
    )
    _ = try await pipeline.generate(request: request, progress: { _ in })

    // Let trailing Tasks flush.
    for _ in 0..<10 { await Task.yield() }

    let events = await processWideRecorder.events

    // Expect at least pipelineStart and pipelineEnd on the process-wide reporter.
    let hasPipelineStart = events.contains {
      if case .pipelineStart = $0 { return true }
      return false
    }
    let hasPipelineEnd = events.contains {
      if case .pipelineEnd = $0 { return true }
      return false
    }

    #expect(
      hasPipelineStart,
      "pipelineStart must reach the process-wide reporter. Events received: \(events.count)"
    )
    #expect(
      hasPipelineEnd,
      "pipelineEnd must reach the process-wide reporter. Events received: \(events.count)"
    )

    // Sanity: denoiseStepStart events should appear too.
    let denoiseStarts = events.filter {
      if case .denoiseStepStart = $0 { return true }
      return false
    }
    #expect(
      denoiseStarts.count == 2,
      "Expected 2 denoiseStepStart events (one per step). Got: \(denoiseStarts.count)"
    )
  }
}
