import Foundation

// MARK: - Telemetry Injection Seam (OPERATION GLASS PIPES Sortie 2)
//
// This extension exposes the public surface for installing a
// `TuberiaTelemetryReporter` on a `DiffusionPipeline`. The reporter itself is
// stored as a private ivar on the actor in `DiffusionPipeline.swift`. All
// emission sites added in later sorties read the ivar through the actor's
// serial executor, so no further synchronization is required at call sites.
//
// The setter is intentionally minimal — no validation, no replay of past
// events. Callers attach exactly once per generation (typically immediately
// after `init(recipe:)`), or pass `nil` to detach. The `nil` case is the
// source-compatible default that preserves the zero-cost-when-off contract
// proven by `TuberiaTelemetryNoopOverheadTests` (Sortie 7).

extension DiffusionPipeline {
  /// Install (or remove) the telemetry reporter that boundary-event emission
  /// sites will dispatch to.
  ///
  /// Pass `nil` to detach; subsequent emission sites short-circuit at the
  /// `if let telemetry { … }` guard and skip the per-tensor `MLX` reductions
  /// that `TuberiaTensorStat.sample(...)` performs. This is the mechanism that
  /// keeps the +1% telemetry-off overhead bound from §7 of the requirements.
  ///
  /// Calling this is the host-side counterpart to the rest of the telemetry
  /// surface: `TuberiaTelemetryReporter` defines the sink, `TuberiaTensorStat`
  /// is the payload, `TuberiaTelemetryEvent` is the discriminator.
  ///
  /// - Parameter reporter: The reporter to attach, or `nil` to detach.
  public func setTelemetry(_ reporter: (any TuberiaTelemetryReporter)?) {
    // Writes through the internal `installTelemetry(_:)` forwarder declared
    // on the actor's body. Direct assignment to `self.telemetry` would not
    // compile here because the ivar is `private` per §4.1 of the
    // instrumentation requirements, and Swift extension access does not
    // cross file boundaries for `private`.
    self.installTelemetry(reporter)
  }
}
