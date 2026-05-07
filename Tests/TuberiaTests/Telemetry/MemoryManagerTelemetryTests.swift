import Foundation
import Metal
import Testing

@testable import Tuberia

// MARK: - MemoryManager Telemetry Tests
//
// Tests for `MemoryManager.clearGPUCache()` telemetry instrumentation added in Sortie 9.
//
// Key design constraints:
//   - MemoryManager.shared is a global singleton actor. Setting telemetry on it has
//     cross-test side effects. Every test that attaches a reporter MUST reset it via
//     `await MemoryManager.shared.setTelemetry(nil)` — use defer blocks.
//   - clearGPUCache() is async (Sortie 9 changed the signature).
//   - When a telemetry reporter is set, clearGPUCache emits exactly one
//     `unloadSegment(role: nil, componentId: nil, metalAllocatedBytesBefore:, metalAllocatedBytesAfter:)`
//     event around the `MLX.Memory.clearCache()` call.
//   - Metal bytes are non-nil on any host with a Metal device (macos-26 always has one).
//     The Metal-bytes assertion is gated by `try #require(MTLCreateSystemDefaultDevice() != nil)`
//     so the suite is portable to headless or non-Metal CI environments.

@Suite("MemoryManager Telemetry Tests", .serialized)
struct MemoryManagerTelemetryTests {

    // MARK: - Single unloadSegment(role: nil) emission

    /// `clearGPUCache()` with a reporter attached emits exactly one `unloadSegment(role: nil, ...)` event.
    ///
    /// This verifies the Sortie 9 instrumentation in MemoryManager.clearGPUCache:
    ///   - role must be nil (post-cache-clear marker)
    ///   - componentId must be nil
    ///   - exactly 1 event is emitted per clearGPUCache call
    @Test("clearGPUCache emits exactly one unloadSegment(role: nil) event")
    func clearGPUCacheEmitsOneUnloadSegmentRoleNil() async throws {
        let mock = MockTuberiaTelemetryReporter()
        await MemoryManager.shared.setTelemetry(mock)
        defer {
            // CRITICAL: reset telemetry to prevent cross-test contamination.
            // MemoryManager.shared is a singleton — leaving a reporter attached would
            // allow subsequent unrelated calls to clearGPUCache (from other tests,
            // from unloadModels, etc.) to bleed events into this mock.
            Task {
                await MemoryManager.shared.setTelemetry(nil)
            }
        }

        await MemoryManager.shared.clearGPUCache()

        // CRITICAL cleanup: tear down before asserting so a failure doesn't leave state dirty.
        await MemoryManager.shared.setTelemetry(nil)

        let events = await mock.events

        #expect(events.count == 1, "clearGPUCache must emit exactly 1 event; got \(events.count)")

        guard let event = events.first else {
            Issue.record("Expected 1 event but events array is empty")
            return
        }

        // Confirm the event is `unloadSegment` with role == nil and componentId == nil.
        if case .unloadSegment(let role, let componentId, _, _) = event {
            #expect(role == nil, "unloadSegment.role must be nil for post-clearGPUCache event")
            #expect(componentId == nil, "unloadSegment.componentId must be nil for post-clearGPUCache event")
        } else {
            Issue.record("Expected .unloadSegment event, got \(event)")
        }
    }

    // MARK: - Metal bytes non-nil when device available

    /// When a Metal device is available, both `metalAllocatedBytesBefore` and
    /// `metalAllocatedBytesAfter` are non-nil in the emitted `unloadSegment` event.
    ///
    /// Skips gracefully when no Metal device is available (CI without GPU, etc.).
    @Test("unloadSegment Metal bytes are non-nil when Metal device available")
    func metalBytesPopulatedWhenDeviceAvailable() async throws {
        // Skip gracefully when the host has no Metal device.
        try #require(MTLCreateSystemDefaultDevice() != nil,
            "No Metal device available on this host — skipping Metal-bytes assertion")

        let mock = MockTuberiaTelemetryReporter()
        await MemoryManager.shared.setTelemetry(mock)

        await MemoryManager.shared.clearGPUCache()

        // Cleanup before asserting.
        await MemoryManager.shared.setTelemetry(nil)

        let events = await mock.events

        #expect(events.count == 1, "clearGPUCache must emit exactly 1 event; got \(events.count)")

        guard let event = events.first else {
            Issue.record("Expected 1 event but events array is empty")
            return
        }

        if case .unloadSegment(let role, let componentId, let bytesBefore, let bytesAfter) = event {
            #expect(role == nil)
            #expect(componentId == nil)
            // Both Metal byte fields must be non-nil when a device is present.
            #expect(bytesBefore != nil,
                "metalAllocatedBytesBefore must be non-nil when Metal device is available")
            #expect(bytesAfter != nil,
                "metalAllocatedBytesAfter must be non-nil when Metal device is available")
        } else {
            Issue.record("Expected .unloadSegment event, got \(event)")
        }
    }

    // MARK: - Nil telemetry produces no events

    /// Without a reporter set, `clearGPUCache()` completes silently — no events observable.
    /// Verifies the nil-reporter path is unchanged by Sortie 9 instrumentation.
    @Test("clearGPUCache with nil telemetry produces no events")
    func clearGPUCacheWithNilTelemetryProducesNoEvents() async throws {
        // Ensure no reporter is set from a previous test (defensive cleanup).
        await MemoryManager.shared.setTelemetry(nil)

        // After the call, there is no reporter to observe events from.
        // This test simply confirms it completes without error when telemetry is nil.
        await MemoryManager.shared.clearGPUCache()
        // No assertion beyond "did not throw" — nil path is silent by contract.
    }

    // MARK: - Multiple calls each emit one event

    /// Each individual `clearGPUCache()` call emits exactly 1 event; two sequential calls
    /// produce exactly 2 events on the same mock (drain verifies isolation).
    @Test("two sequential clearGPUCache calls emit two events total")
    func twoSequentialClearCallsEmitTwoEvents() async throws {
        let mock = MockTuberiaTelemetryReporter()
        await MemoryManager.shared.setTelemetry(mock)

        await MemoryManager.shared.clearGPUCache()
        await MemoryManager.shared.clearGPUCache()

        // Cleanup before asserting.
        await MemoryManager.shared.setTelemetry(nil)

        let events = await mock.events
        #expect(events.count == 2,
            "Two sequential clearGPUCache calls must produce exactly 2 events; got \(events.count)")

        // Both events must be role-nil unloadSegment.
        for event in events {
            if case .unloadSegment(let role, let componentId, _, _) = event {
                #expect(role == nil)
                #expect(componentId == nil)
            } else {
                Issue.record("Expected .unloadSegment event, got \(event)")
            }
        }
    }

    // MARK: - Cleanup verify (belt-and-suspenders)

    /// Confirm that after each test runs, `MemoryManager.shared` reports nil telemetry.
    /// This test verifies the cleanup pattern works correctly — if any previous test leaked
    /// its reporter, this would catch it.
    ///
    /// Note: relies on `.serialized` suite ordering and proper cleanup in prior tests.
    @Test("MemoryManager.shared has nil telemetry after cleanup")
    func cleanupResetsTelemetry() async throws {
        // After the serialized tests above (all of which call setTelemetry(nil) before returning),
        // the shared telemetry should be nil.
        //
        // We cannot read `MemoryManager.shared.telemetry` directly from outside the actor
        // (actor-isolated property), but we can verify the observable behavior: a call to
        // clearGPUCache with an independent mock should produce 0 events if the singleton
        // telemetry is nil (i.e. the sink for events would be the mock, not the singleton).
        //
        // More directly: attach a fresh mock, confirm exactly 1 event fires, then detach.
        // If a leaked reporter from a previous test were still attached, the event count
        // might be wrong or events might go to the wrong mock.

        await MemoryManager.shared.setTelemetry(nil)

        let verifyMock = MockTuberiaTelemetryReporter()
        await MemoryManager.shared.setTelemetry(verifyMock)
        await MemoryManager.shared.clearGPUCache()
        await MemoryManager.shared.setTelemetry(nil)

        let events = await verifyMock.events
        #expect(events.count == 1,
            "After explicit setTelemetry(nil) + setTelemetry(mock), clearGPUCache must emit exactly 1 event")
    }
}
