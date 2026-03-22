import Foundation
import MLX

#if canImport(Darwin)
import Darwin
import MachO
#endif

/// Coordinates memory across all loaded pipe segments. Global singleton actor.
///
/// MemoryManager tracks all loaded components across all pipelines (image, TTS, etc.).
/// It reports total loaded memory but does NOT auto-unload -- the caller/app decides priority.
/// If budget is tight, MemoryManager returns `false`/throws and the caller decides what to evict.
///
/// Headroom multipliers are per-consumer (applied externally), not in MemoryManager.
public actor MemoryManager {

    /// Global singleton instance.
    public static let shared = MemoryManager()

    /// Tracks loaded component sizes by component ID.
    private var loadedComponents: [String: UInt64] = [:]

    private init() {}

    // MARK: - Device Capability

    /// Returns the same value as `DeviceCapability.current`.
    /// Provided for contexts that already have an actor reference.
    public var deviceCapability: DeviceCapability {
        DeviceCapability.current
    }

    // MARK: - Memory Queries

    /// Total physical memory in bytes.
    public var totalMemory: UInt64 {
        var memSize: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &memSize, &size, nil, 0)
        return memSize
    }

    /// Available memory in bytes, using Mach VM statistics.
    /// Includes free + inactive + purgeable + speculative pages for a realistic
    /// picture of usable memory rather than just "free" pages.
    public var availableMemory: UInt64 {
        #if canImport(Darwin)
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size
        )

        let result = withUnsafeMutablePointer(to: &stats) { statsPtr in
            statsPtr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { ptr in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, ptr, &count)
            }
        }

        guard result == KERN_SUCCESS else {
            // Fallback: return total memory minus a conservative estimate
            return totalMemory / 2
        }

        let pageSize = UInt64(getpagesize())
        let free = UInt64(stats.free_count) * pageSize
        let inactive = UInt64(stats.inactive_count) * pageSize
        let purgeable = UInt64(stats.purgeable_count) * pageSize
        let speculative = UInt64(stats.speculative_count) * pageSize

        return free + inactive + purgeable + speculative
        #else
        return totalMemory / 2
        #endif
    }

    // MARK: - Budget Checks

    /// Soft check: returns `true` if available memory exceeds the requirement.
    /// Does not throw. Callers use this to decide loading strategy (eager vs. phased).
    public func softCheck(requiredBytes: UInt64) -> Bool {
        availableMemory >= requiredBytes
    }

    /// Hard validation: throws `PipelineError.insufficientMemory` if the budget is exceeded.
    /// Use this as a gate before committing to a load operation.
    public func hardValidate(requiredBytes: UInt64) throws {
        let available = availableMemory
        guard available >= requiredBytes else {
            throw PipelineError.insufficientMemory(
                required: requiredBytes,
                available: available,
                component: "pipeline"
            )
        }
    }

    // MARK: - Component Tracking

    /// Register a loaded component and its memory footprint.
    public func registerLoaded(component: String, bytes: UInt64) {
        loadedComponents[component] = bytes
    }

    /// Unregister a component that has been unloaded.
    public func unregisterLoaded(component: String) {
        loadedComponents.removeValue(forKey: component)
    }

    /// Total memory in bytes consumed by all currently loaded components.
    public var loadedComponentsMemory: UInt64 {
        loadedComponents.values.reduce(0, +)
    }

    // MARK: - GPU Cache

    /// Clear the MLX GPU buffer cache.
    /// Call between loading phases to free memory for the next phase.
    public func clearGPUCache() {
        MLX.Memory.clearCache()
    }
}
