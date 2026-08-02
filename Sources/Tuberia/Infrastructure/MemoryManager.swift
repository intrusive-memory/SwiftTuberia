import Foundation
import MLX

#if canImport(Darwin)
  import Darwin
  import MachO
#endif

// `os_proc_available_memory()` is declared in `os/proc.h`, which Swift surfaces
// through the `os` module rather than `Darwin`.
#if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
  import os
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

  /// Memory this process may still allocate, in bytes.
  ///
  /// **The number this returns is deliberately different per platform, because
  /// the thing that kills you is different per platform.**
  ///
  /// On iOS/tvOS/watchOS/visionOS the limit is *per process*: jetsam watches
  /// this task's `phys_footprint` against a cap that is a fraction of device
  /// RAM, and kills the app when it crosses. System-wide free memory says
  /// nothing about how close you are to that cap — a device can report
  /// gigabytes free while this process is megabytes from being killed. So we
  /// ask `os_proc_available_memory()`, which reports exactly the remaining
  /// headroom before this process hits its own limit.
  ///
  /// On macOS there is no such per-process cap for ordinary apps; the kernel
  /// reclaims from compressed, inactive, and cached pages and swaps on demand.
  /// There, system-wide reclaimable memory (free + inactive + purgeable +
  /// speculative) is the meaningful figure and `os_proc_available_memory()` is
  /// unavailable.
  ///
  /// Consumers should not special-case platforms themselves — ``softCheck(requiredBytes:)``
  /// and ``hardValidate(requiredBytes:telemetry:)`` are built on this and are
  /// correct on both.
  public var availableMemory: UInt64 {
    #if os(iOS) || os(tvOS) || os(watchOS) || os(visionOS)
      // Remaining bytes before THIS process trips its jetsam footprint limit.
      // Returns 0 if unavailable, which we treat as "no headroom left" rather
      // than silently falling back to a system-wide number that would answer a
      // different question.
      return UInt64(max(0, os_proc_available_memory()))
    #elseif canImport(Darwin)
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
  ///
  /// - Parameters:
  ///   - requiredBytes: The memory budget to validate against.
  ///   - telemetry: Optional telemetry reporter forwarded from
  ///     `DiffusionPipeline.memoryGate`. Defaults to `nil` so existing call
  ///     sites compile unchanged. Sortie 3+ will wire `memoryGateChecked` /
  ///     `errorThrown` against this parameter.
  public func hardValidate(
    requiredBytes: UInt64,
    telemetry: (any TuberiaTelemetryReporter)? = nil
  ) throws {
    // Plumbing only — no emission yet (Sortie 2).
    _ = telemetry
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

  // MARK: - Resident Footprint

  /// Current resident footprint (`phys_footprint`) of this process, in bytes.
  ///
  /// This is the number Jetsam / the memory limit actually watches, so it is the
  /// right signal for "did the decode spike toward the cap?" (#45). Read-only and
  /// cheap (microseconds). Returns 0 if the `task_info` call fails or on a
  /// non-Darwin platform.
  public var residentFootprint: UInt64 {
    #if canImport(Darwin)
      var info = task_vm_info_data_t()
      var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
      )
      let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
          task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
      }
      guard kr == KERN_SUCCESS else { return 0 }
      return UInt64(info.phys_footprint)
    #else
      return 0
    #endif
  }

  // MARK: - GPU Cache

  /// Clear the MLX GPU buffer cache.
  /// Call between loading phases to free memory for the next phase.
  public func clearGPUCache() {
    MLX.Memory.clearCache()
  }
}
