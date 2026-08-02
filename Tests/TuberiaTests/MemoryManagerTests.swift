import Testing

@testable import Tuberia

/// Unit tests for MemoryManager budget tracking and validation.
///
/// Note: MemoryManager.shared is a global singleton. Tests use unique component names
/// and test relative changes rather than absolute values to avoid interference
/// from concurrent test execution.
@Suite("MemoryManager Tests", .serialized)
struct MemoryManagerTests {

  // MARK: - Component Tracking

  @Test("register, overwrite, and unregister track bytes accurately")
  func componentTrackingLifecycle() async throws {
    let manager = MemoryManager.shared

    let compA = "track-\(UInt64.random(in: 0...UInt64.max))"
    let compB = "track-\(UInt64.random(in: 0...UInt64.max))"

    let baseline = await manager.loadedComponentsMemory

    await manager.registerLoaded(component: compA, bytes: 1_000_000)
    #expect(await manager.loadedComponentsMemory - baseline == 1_000_000)

    await manager.registerLoaded(component: compB, bytes: 2_000_000)
    #expect(await manager.loadedComponentsMemory - baseline == 3_000_000)

    // Overwrite compA with different size
    await manager.registerLoaded(component: compA, bytes: 5_000)
    #expect(await manager.loadedComponentsMemory - baseline == 2_005_000)

    await manager.unregisterLoaded(component: compA)
    #expect(await manager.loadedComponentsMemory - baseline == 2_000_000)

    await manager.unregisterLoaded(component: compB)
    #expect(await manager.loadedComponentsMemory == baseline)
  }

  @Test("unregisterLoaded is idempotent for unknown components")
  func unregisterUnknownComponent() async throws {
    let manager = MemoryManager.shared
    let comp = "nonexistent-\(UInt64.random(in: 0...UInt64.max))"

    let before = await manager.loadedComponentsMemory
    await manager.unregisterLoaded(component: comp)
    let after = await manager.loadedComponentsMemory
    #expect(before == after)
  }

  // MARK: - Budget Checks

  @Test("softCheck: true for small request, false for UInt64.max")
  func softCheckBoundaries() async throws {
    let manager = MemoryManager.shared
    #expect(await manager.softCheck(requiredBytes: 1) == true)
    #expect(await manager.softCheck(requiredBytes: UInt64.max) == false)
  }

  @Test("hardValidate: passes for tiny request, throws for impossible request")
  func hardValidateBoundaries() async throws {
    let manager = MemoryManager.shared

    try await manager.hardValidate(requiredBytes: 1)

    do {
      try await manager.hardValidate(requiredBytes: UInt64.max)
      Issue.record("Expected PipelineError.insufficientMemory")
    } catch let error as PipelineError {
      switch error {
      case .insufficientMemory(let required, let available, _):
        #expect(required == UInt64.max)
        #expect(available < UInt64.max)
      default:
        Issue.record("Expected insufficientMemory, got \(error)")
      }
    }
  }

  // MARK: - Memory Queries

  @Test("totalMemory and availableMemory are positive")
  func memoryQueriesPositive() async throws {
    let manager = MemoryManager.shared
    #expect(await manager.totalMemory > 0)
    #expect(await manager.availableMemory > 0)
  }

  @Test("availableMemory never exceeds total physical memory")
  func availableMemoryIsBoundedByTotal() async throws {
    // Holds on both platforms, but it is the iOS reading this really guards:
    // there `availableMemory` is `os_proc_available_memory()` — headroom before
    // THIS process trips its jetsam footprint cap — which is a fraction of
    // device RAM. A value at or above `totalMemory` would mean the per-process
    // gate had silently reverted to a system-wide sum, which is the bug this
    // API exists to prevent.
    let manager = MemoryManager.shared
    let total = await manager.totalMemory
    let available = await manager.availableMemory
    #expect(available <= total)
  }

  @Test("residentFootprint reports this process's phys_footprint")
  func residentFootprintIsPositiveAndBounded() async throws {
    // `residentFootprint` is the number jetsam actually watches. It was already
    // present but unused by any gate; assert it is live so it stays wired.
    let manager = MemoryManager.shared
    let footprint = await manager.residentFootprint
    #expect(footprint > 0)
    #expect(footprint <= (await manager.totalMemory))
  }

  // MARK: - Device Capability

  @Test("deviceCapability matches DeviceCapability.current")
  func deviceCapabilityConsistency() async throws {
    let manager = MemoryManager.shared
    let fromManager = await manager.deviceCapability
    let direct = DeviceCapability.current

    #expect(fromManager.totalMemoryGB == direct.totalMemoryGB)
    #expect(fromManager.platform == direct.platform)
    #expect(fromManager.chipGeneration == direct.chipGeneration)
  }

  // MARK: - GPU Cache

  @Test("clearGPUCache completes without error")
  func clearGPUCacheNoError() async throws {
    let manager = MemoryManager.shared
    await manager.clearGPUCache()
  }

  // MARK: - Resident Footprint

  #if canImport(Darwin)
    @Test("residentFootprint returns a positive byte count on Darwin")
    func residentFootprintPositive() async throws {
      let manager = MemoryManager.shared
      let footprint = await manager.residentFootprint
      // task_info(TASK_VM_INFO) is expected to succeed in-process; a running
      // process always has a non-zero phys_footprint.
      #expect(footprint > 0)
    }
  #endif
}
