import Foundation
import Testing
import Tuberia

/// Proves that `TuberiaTensorStat` is publicly importable, `Codable`, and
/// `Sendable` from outside the `Tuberia` target.
///
/// Downstream MLX libraries (`flux-2-swift-mlx`, `pixart-swift-mlx`,
/// `SwiftVinetas`) `import Tuberia` and consume this type — so the import is
/// non-`@testable` on purpose. If anything in `TuberiaTensorStat`'s public
/// surface changes shape, this test fails to compile and the cross-repo ABI
/// break is caught here rather than in a downstream CI.
@Suite("TuberiaTensorStat public API")
struct TuberiaTensorStatPublicAPITests {

  @Test("Round-trips through JSONEncoder/JSONDecoder with every field preserved")
  func roundTripsThroughJSON() throws {
    let original = TuberiaTensorStat(
      shape: [1],
      dtype: "float32",
      min: 0,
      max: 0,
      mean: 0,
      std: 0,
      hasNaN: false,
      hasInf: false
    )

    let data = try JSONEncoder().encode(original)
    let decoded = try JSONDecoder().decode(TuberiaTensorStat.self, from: data)

    #expect(decoded.shape == original.shape)
    #expect(decoded.dtype == original.dtype)
    #expect(decoded.min == original.min)
    #expect(decoded.max == original.max)
    #expect(decoded.mean == original.mean)
    #expect(decoded.std == original.std)
    #expect(decoded.hasNaN == original.hasNaN)
    #expect(decoded.hasInf == original.hasInf)
    #expect(decoded == original)
  }

  @Test("Round-trips a multi-dim, populated stat with NaN/Inf flags set")
  func roundTripsPopulatedStat() throws {
    let original = TuberiaTensorStat(
      shape: [2, 3, 4, 5],
      dtype: "bfloat16",
      min: -123.456,
      max: 789.012,
      mean: 0.5,
      std: 1.25,
      hasNaN: true,
      hasInf: true
    )

    let data = try JSONEncoder().encode(original)
    let decoded = try JSONDecoder().decode(TuberiaTensorStat.self, from: data)

    #expect(decoded == original)
    #expect(decoded.shape == [2, 3, 4, 5])
    #expect(decoded.dtype == "bfloat16")
    #expect(decoded.hasNaN == true)
    #expect(decoded.hasInf == true)
  }

  @Test("Default out-of-range threshold is 1e6 (Q2 resolution)")
  func defaultOutOfRangeThresholdIs1e6() {
    #expect(TuberiaTensorStat.defaultOutOfRangeThreshold == 1e6)
  }
}
