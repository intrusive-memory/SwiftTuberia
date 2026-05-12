import Foundation

/// Sink protocol for `TuberiaTelemetryEvent`s.
///
/// Conformance is `Sendable` so a single reporter instance can be stored on
/// the `DiffusionPipeline` actor and read from every emission site without
/// further synchronization. The single requirement is `async` because the
/// Vinetas-host adapter typically forwards events into an actor-backed sink
/// and we do not want the wrapper to swallow the await.
///
/// **No `@autoclosure` on `capture(_:)`.** Hot-path emission cost is gated at
/// each call site with `if let telemetry { … }`. Pushing the guard into the
/// protocol would force every caller through an indirection even when the
/// reporter is `nil`, defeating the zero-cost-when-off contract proven by the
/// `TuberiaTelemetryNoopOverheadTests` overhead bound.
public protocol TuberiaTelemetryReporter: Sendable {
  /// Capture a single telemetry event. Implementations decide whether to
  /// buffer, forward, drop, or block — the call site only awaits the hop.
  func capture(_ event: TuberiaTelemetryEvent) async
}

/// A `TuberiaTelemetryReporter` that discards every event.
///
/// Used by the overhead test as the "telemetry-on but doing nothing" baseline:
/// callers still construct events and pay one `async` hop per emission, which
/// is the worst-case shape for downstream libraries that wire a reporter
/// without actually consuming events. The +1% overhead bar (see
/// `TuberiaTelemetryNoopOverheadTests`) is measured against this struct, not
/// against a `nil` reporter.
public struct NoopTuberiaTelemetryReporter: TuberiaTelemetryReporter {
  public init() {}
  public func capture(_ event: TuberiaTelemetryEvent) async {}
}
