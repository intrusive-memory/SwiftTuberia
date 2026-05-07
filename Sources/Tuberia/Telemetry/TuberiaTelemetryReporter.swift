/// Receives telemetry events from SwiftTuberia. Implementations forward events to
/// whatever sink the host application uses (Produciesta's MemoryTelemetry actor,
/// OSLog, Sentry, swift-metrics, custom JSONL writer, etc).
///
/// SwiftTuberia does not depend on, log to, or assume the existence of any specific
/// telemetry framework. If no reporter is set, no events are emitted.
///
/// ## Concurrency
///
/// `Sendable`-bounded so the pipeline (an `actor`) can hold and call across hops.
/// `capture` is `async` so a reporter that batches or writes to disk can do so
/// without blocking pipeline progress, but it must remain non-throwing — the
/// pipeline never lets telemetry fail a generation.
public protocol TuberiaTelemetryReporter: Sendable {
  func capture(_ event: TuberiaTelemetryEvent) async
}
