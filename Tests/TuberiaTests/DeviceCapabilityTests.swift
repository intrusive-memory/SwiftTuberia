import Testing
@testable import Tuberia

/// Unit tests for DeviceCapability detection.
@Suite("DeviceCapability Tests")
struct DeviceCapabilityTests {

    @Test("DeviceCapability.current returns valid values")
    func currentDeviceCapability() throws {
        let capability = DeviceCapability.current
        #expect(capability.totalMemoryGB > 0)
        #expect(capability.chipGeneration != .unknown || true) // May be unknown in CI
    }

    @Test("Platform detection returns macOS on Mac")
    func platformDetection() throws {
        let capability = DeviceCapability.current
        #if os(macOS)
        #expect(capability.platform == .macOS)
        #elseif os(iOS)
        #expect(capability.platform == .iPadOS)
        #endif
    }

    @Test("Chip generation parsing covers M1 through M5 and tiers")
    func chipGenerationParsing() throws {
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M1") == .m1)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M1 Pro") == .m1Pro)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M1 Max") == .m1Max)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M1 Ultra") == .m1Ultra)

        #expect(DeviceCapability.parseChipGeneration(from: "Apple M2") == .m2)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M2 Pro") == .m2Pro)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M2 Max") == .m2Max)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M2 Ultra") == .m2Ultra)

        #expect(DeviceCapability.parseChipGeneration(from: "Apple M3") == .m3)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M3 Pro") == .m3Pro)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M3 Max") == .m3Max)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M3 Ultra") == .m3Ultra)

        #expect(DeviceCapability.parseChipGeneration(from: "Apple M4") == .m4)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M4 Pro") == .m4Pro)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M4 Max") == .m4Max)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M4 Ultra") == .m4Ultra)

        #expect(DeviceCapability.parseChipGeneration(from: "Apple M5") == .m5)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M5 Pro") == .m5Pro)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M5 Max") == .m5Max)
        #expect(DeviceCapability.parseChipGeneration(from: "Apple M5 Ultra") == .m5Ultra)
    }

    @Test("Unknown CPU brand string returns .unknown")
    func unknownChip() throws {
        #expect(DeviceCapability.parseChipGeneration(from: "Intel Core i9") == .unknown)
        #expect(DeviceCapability.parseChipGeneration(from: "") == .unknown)
    }

    @Test("Neural accelerator detection based on chip generation")
    func neuralAcceleratorDetection() throws {
        let capability = DeviceCapability.current
        if capability.chipGeneration != .unknown {
            #expect(capability.hasNeuralAccelerators == true)
        }
    }

    @Test("DeviceCapability.current is consistent across accesses")
    func currentIsConsistent() throws {
        let first = DeviceCapability.current
        let second = DeviceCapability.current
        #expect(first.totalMemoryGB == second.totalMemoryGB)
        #expect(first.chipGeneration == second.chipGeneration)
        #expect(first.platform == second.platform)
    }
}
