import Testing
@testable import Tuberia

/// Actor-isolated telemetry recorder for use in TuberiaTests.
///
/// Conforms to `TuberiaTelemetryReporter` and records every captured event in order.
/// Reused across Sorties 2, 4, 7, 8, and 10.
///
/// ## Usage
///
/// ```swift
/// let mock = MockTuberiaTelemetryReporter()
/// await pipeline.setTelemetry(mock)
/// try await pipeline.loadModels { _, _ in }
/// let events = await mock.events
/// ```
actor MockTuberiaTelemetryReporter: TuberiaTelemetryReporter {

  private var _events: [TuberiaTelemetryEvent] = []

  // MARK: - TuberiaTelemetryReporter

  func capture(_ event: TuberiaTelemetryEvent) async {
    _events.append(event)
  }

  // MARK: - Accessors

  /// All events recorded so far, in capture order.
  var events: [TuberiaTelemetryEvent] {
    _events
  }

  /// Returns a snapshot of all recorded events and then clears the buffer.
  /// Useful when a single mock is reused across multiple pipeline invocations.
  func drain() -> [TuberiaTelemetryEvent] {
    let snapshot = _events
    _events = []
    return snapshot
  }
}
