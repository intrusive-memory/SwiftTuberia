import Foundation

@testable import Tuberia

// MARK: - RecordingTelemetryReporter
//
// Actor-based `TuberiaTelemetryReporter` used by Sortie 6 functional tests.
// Appends every captured event to an internal `[TuberiaTelemetryEvent]` in
// emission order. Actor isolation prevents data races when the DiffusionPipeline
// actor emits events asynchronously.

public actor RecordingTelemetryReporter: TuberiaTelemetryReporter {
  private var _events: [TuberiaTelemetryEvent] = []

  public init() {}

  // MARK: - TuberiaTelemetryReporter

  public func capture(_ event: TuberiaTelemetryEvent) async {
    _events.append(event)
  }

  // MARK: - Accessors

  /// Snapshot of all captured events in emission order.
  public var events: [TuberiaTelemetryEvent] {
    _events
  }

  /// Reset captured events (call between test cases for isolation).
  public func clear() {
    _events = []
  }
}

// MARK: - Convenience helpers (non-isolated, caller must be in async context)

extension RecordingTelemetryReporter {
  /// Extract and return all events matching a given pattern.
  ///
  /// Usage:
  /// ```swift
  /// let starts = await rec.eventsMatching {
  ///   if case .denoiseStepStart(let idx, _, _, _, _, _) = $0 { return idx } else { return nil }
  /// }
  /// ```
  public func eventsMatching<T: Sendable>(
    _ extract: @Sendable (TuberiaTelemetryEvent) -> T?
  ) -> [T] {
    _events.compactMap(extract)
  }
}
