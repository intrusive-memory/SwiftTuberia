import os.lock

// MARK: - Process-wide Telemetry Seam
//
// `TuberiaTelemetry` provides a process-wide reporter shared by all current and
// future `DiffusionPipeline` instances. CLI hosts (e.g. SwiftVinetas) that own no
// direct reference to a pipeline can call `TuberiaTelemetry.setReporter(_:)` once
// at process startup; all Tuberia emission sites then route events there whenever
// no instance-specific reporter is installed.
//
// **Priority rule**: the instance-bound reporter set via
// `DiffusionPipeline.setTelemetry(_:)` always wins over the process-wide
// reporter. The process-wide reporter fires only when the instance reporter is
// `nil`. This is enforced by every emission site reading `effectiveReporter`
// (or the equivalent computed property) rather than consulting the two reporters
// independently.
//
// **Thread safety**: `OSAllocatedUnfairLock` gives a lock-free read path on the
// fast-path after the first store; the initial `nil` state means zero contention
// until a reporter is installed.
//
// **Reset semantics**: `TuberiaTelemetry.setReporter(nil)` clears the
// process-wide reporter and restores the "no reporter" state.

/// Process-wide telemetry seam for SwiftTuberia.
///
/// Install a reporter once at process startup so that every current and future
/// `DiffusionPipeline` instance emits events to it — without requiring a
/// per-instance `setTelemetry(_:)` call.
///
/// ```swift
/// // At CLI launch:
/// TuberiaTelemetry.setReporter(MyAdapter())
/// ```
///
/// The instance-bound reporter installed via `DiffusionPipeline.setTelemetry(_:)`
/// always takes priority over this process-wide reporter.
public enum TuberiaTelemetry {
    private static let _lock = OSAllocatedUnfairLock<(any TuberiaTelemetryReporter)?>(
        initialState: nil
    )

    /// Install (or clear) the process-wide telemetry reporter.
    ///
    /// Thread-safe — may be called from any thread or actor. Pass `nil` to
    /// detach the current reporter and restore "no reporter" state.
    public static func setReporter(_ reporter: (any TuberiaTelemetryReporter)?) {
        _lock.withLock { $0 = reporter }
    }

    /// The currently installed process-wide reporter, or `nil` if none has been set.
    ///
    /// Emission sites call this on every event via `effectiveReporter` (the
    /// computed property that applies the instance-wins-over-process-wide
    /// priority rule). Reading this property is wait-free in the uncontended
    /// steady state.
    public static var current: (any TuberiaTelemetryReporter)? {
        _lock.withLock { $0 }
    }
}
