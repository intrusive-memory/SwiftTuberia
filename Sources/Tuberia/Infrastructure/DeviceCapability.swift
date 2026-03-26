import Foundation

#if canImport(Darwin)
  import Darwin
#endif

/// Describes the hardware capabilities of the current Apple Silicon device.
/// Cached at first access. Use `DeviceCapability.current` for synchronous access.
public struct DeviceCapability: Sendable {

  /// The Apple Silicon chip generation and tier.
  public let chipGeneration: AppleSiliconGeneration

  /// Total physical memory in gigabytes.
  public let totalMemoryGB: Int

  /// The platform this code is running on.
  public let platform: Platform

  /// Whether Neural Engine accelerators are available.
  public let hasNeuralAccelerators: Bool

  /// Synchronous static accessor. Cached at first access. No actor required.
  /// This is the recommended accessor for all consumers.
  public static let current: DeviceCapability = DeviceCapability.detect()

  // MARK: - AppleSiliconGeneration

  public enum AppleSiliconGeneration: String, Sendable, CaseIterable {
    case m1, m1Pro, m1Max, m1Ultra
    case m2, m2Pro, m2Max, m2Ultra
    case m3, m3Pro, m3Max, m3Ultra
    case m4, m4Pro, m4Max, m4Ultra
    case m5, m5Pro, m5Max, m5Ultra
    case unknown
  }

  // MARK: - Platform

  public enum Platform: String, Sendable {
    case macOS, iPadOS
  }

  // MARK: - Private Detection

  private static func detect() -> DeviceCapability {
    let chipGeneration = detectChipGeneration()
    let totalMemoryGB = detectTotalMemoryGB()
    let platform = detectPlatform()
    let hasNeuralAccelerators = detectNeuralAccelerators(chipGeneration)

    return DeviceCapability(
      chipGeneration: chipGeneration,
      totalMemoryGB: totalMemoryGB,
      platform: platform,
      hasNeuralAccelerators: hasNeuralAccelerators
    )
  }

  private static func detectChipGeneration() -> AppleSiliconGeneration {
    let brandString = sysctlString("machdep.cpu.brand_string") ?? ""
    return parseChipGeneration(from: brandString)
  }

  /// Parse the chip generation from the CPU brand string.
  /// Examples: "Apple M1", "Apple M2 Pro", "Apple M3 Max", "Apple M4 Ultra"
  internal static func parseChipGeneration(from brandString: String) -> AppleSiliconGeneration {
    let lowered = brandString.lowercased()

    // Match patterns like "apple m1", "apple m2 pro", etc.
    // Order matters: check longer suffixes first (Ultra, Max, Pro) before base.
    let generations: [(prefix: String, suffix: String, generation: AppleSiliconGeneration)] = [
      ("m5", "ultra", .m5Ultra), ("m5", "max", .m5Max), ("m5", "pro", .m5Pro),
      ("m4", "ultra", .m4Ultra), ("m4", "max", .m4Max), ("m4", "pro", .m4Pro),
      ("m3", "ultra", .m3Ultra), ("m3", "max", .m3Max), ("m3", "pro", .m3Pro),
      ("m2", "ultra", .m2Ultra), ("m2", "max", .m2Max), ("m2", "pro", .m2Pro),
      ("m1", "ultra", .m1Ultra), ("m1", "max", .m1Max), ("m1", "pro", .m1Pro),
    ]

    for entry in generations {
      if lowered.contains(entry.prefix) && lowered.contains(entry.suffix) {
        return entry.generation
      }
    }

    // Base models (no tier suffix)
    let baseModels: [(prefix: String, generation: AppleSiliconGeneration)] = [
      ("m5", .m5), ("m4", .m4), ("m3", .m3), ("m2", .m2), ("m1", .m1),
    ]

    for entry in baseModels {
      if lowered.contains(entry.prefix) {
        return entry.generation
      }
    }

    return .unknown
  }

  private static func detectTotalMemoryGB() -> Int {
    var memSize: UInt64 = 0
    var size = MemoryLayout<UInt64>.size
    sysctlbyname("hw.memsize", &memSize, &size, nil, 0)
    return Int(memSize / (1024 * 1024 * 1024))
  }

  private static func detectPlatform() -> Platform {
    #if os(macOS)
      return .macOS
    #elseif os(iOS)
      return .iPadOS
    #else
      return .macOS
    #endif
  }

  private static func detectNeuralAccelerators(_ chip: AppleSiliconGeneration) -> Bool {
    // All Apple Silicon chips (M1+) have Neural Engine.
    // Only return false for truly unknown hardware.
    return chip != .unknown
  }

  private static func sysctlString(_ name: String) -> String? {
    var size: Int = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else {
      return nil
    }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else {
      return nil
    }
    return buffer.withUnsafeBufferPointer { bufferPtr in
      bufferPtr.baseAddress.map { ptr in
        String(validatingCString: ptr) ?? ""
      } ?? ""
    }
  }
}
