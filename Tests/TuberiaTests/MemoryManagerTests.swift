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

    @Test("registerLoaded and unregisterLoaded track bytes accurately")
    func componentTracking() async throws {
        let manager = MemoryManager.shared

        let compA = "track-\(UInt64.random(in: 0...UInt64.max))"
        let compB = "track-\(UInt64.random(in: 0...UInt64.max))"

        let baseline = await manager.loadedComponentsMemory

        await manager.registerLoaded(component: compA, bytes: 1_000_000)
        let afterA = await manager.loadedComponentsMemory
        #expect(afterA - baseline == 1_000_000)

        await manager.registerLoaded(component: compB, bytes: 2_000_000)
        let afterAB = await manager.loadedComponentsMemory
        #expect(afterAB - baseline == 3_000_000)

        await manager.unregisterLoaded(component: compA)
        let afterUnloadA = await manager.loadedComponentsMemory
        #expect(afterUnloadA - baseline == 2_000_000)

        await manager.unregisterLoaded(component: compB)
        let afterUnloadAll = await manager.loadedComponentsMemory
        #expect(afterUnloadAll == baseline)
    }

    @Test("registerLoaded overwrites existing entry for same component")
    func componentTrackingOverwrite() async throws {
        let manager = MemoryManager.shared
        let comp = "overwrite-\(UInt64.random(in: 0...UInt64.max))"

        let baseline = await manager.loadedComponentsMemory

        await manager.registerLoaded(component: comp, bytes: 1_000)
        let after1 = await manager.loadedComponentsMemory
        #expect(after1 - baseline == 1_000)

        // Re-register with different size should overwrite
        await manager.registerLoaded(component: comp, bytes: 5_000)
        let after2 = await manager.loadedComponentsMemory
        #expect(after2 - baseline == 5_000)

        // Cleanup
        await manager.unregisterLoaded(component: comp)
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

    // MARK: - Soft Check

    @Test("softCheck returns true when enough memory is available")
    func softCheckPasses() async throws {
        let manager = MemoryManager.shared
        // Request a very small amount that should always be available
        let result = await manager.softCheck(requiredBytes: 1)
        #expect(result == true)
    }

    @Test("softCheck returns false when budget would be exceeded")
    func softCheckFails() async throws {
        let manager = MemoryManager.shared
        // Request more memory than any machine has
        let result = await manager.softCheck(requiredBytes: UInt64.max)
        #expect(result == false)
    }

    // MARK: - Hard Validate

    @Test("hardValidate passes with reasonable memory requirement")
    func hardValidatePasses() async throws {
        let manager = MemoryManager.shared
        // Request a tiny amount
        try await manager.hardValidate(requiredBytes: 1)
        // If we reach here, no throw = success
    }

    @Test("hardValidate throws insufficientMemory when budget exceeded")
    func hardValidateThrows() async throws {
        let manager = MemoryManager.shared
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

    @Test("totalMemory returns a positive value")
    func totalMemoryPositive() async throws {
        let manager = MemoryManager.shared
        let total = await manager.totalMemory
        #expect(total > 0)
    }

    @Test("availableMemory returns a positive value")
    func availableMemoryPositive() async throws {
        let manager = MemoryManager.shared
        let available = await manager.availableMemory
        #expect(available > 0)
    }

    // MARK: - Device Capability

    @Test("deviceCapability returns same value as DeviceCapability.current")
    func deviceCapabilityConsistency() async throws {
        let manager = MemoryManager.shared
        let fromManager = await manager.deviceCapability
        let direct = DeviceCapability.current

        #expect(fromManager.totalMemoryGB == direct.totalMemoryGB)
        #expect(fromManager.platform == direct.platform)
        #expect(fromManager.chipGeneration == direct.chipGeneration)
    }

    // MARK: - GPU Cache Clear

    @Test("clearGPUCache completes without error")
    func clearGPUCacheNoError() async throws {
        let manager = MemoryManager.shared
        await manager.clearGPUCache()
        // If we reach here, no crash = success
    }
}
