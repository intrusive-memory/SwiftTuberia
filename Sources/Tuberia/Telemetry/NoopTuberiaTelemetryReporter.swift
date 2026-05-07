/// A no-op telemetry reporter that discards all events.
///
/// Use this when you want to attach a reporter without incurring any overhead
/// beyond the nil-check and async hop — useful in tests that verify the pipeline
/// runs without telemetry-related crashes.
public struct NoopTuberiaTelemetryReporter: TuberiaTelemetryReporter {
  public init() {}
  public func capture(_ event: TuberiaTelemetryEvent) async {}
}
