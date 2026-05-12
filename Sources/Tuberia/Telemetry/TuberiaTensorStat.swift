import Foundation
@preconcurrency import MLX

/// Per-tensor statistics payload published across the SwiftTuberia → flux-2 →
/// pixart → SwiftVinetas dependency graph.
///
/// `TuberiaTensorStat` is the canonical handoff record: every protocol-boundary
/// telemetry event carries one (or two) of these for the tensor that just
/// crossed the seam. Downstream MLX libraries (`flux-2-swift-mlx`,
/// `pixart-swift-mlx`, `SwiftVinetas`) `import Tuberia` and consume this type
/// without redefining it.
///
/// **Cost model.** Sampling a tensor runs eight MLX reductions
/// (`min`, `max`, `mean`, `std`, `isNaN.any`, `isInf.any`) plus a single
/// `eval()` to materialize them in one Metal command-buffer. Callers MUST
/// guard the call with `if let telemetry { ... }` — `sample(_:)` performs no
/// telemetry side-effects of its own (see Q4 of the GLASS PIPES execution
/// plan), so the only way to get zero-cost-when-off semantics is to keep
/// `sample(_:)` out of the hot path entirely when telemetry is `nil`.
///
/// Numeric outputs are widened to `Double` so a float16 sample produces the
/// same struct shape (and the same JSON encoding) as a float32 sample. The
/// `dtype` string is the canonical lowercase MLX name (`"float16"`,
/// `"float32"`, `"bfloat16"`, `"int32"`, …) used by the Vinetas-host adapter
/// to derive its phase suffix.
public struct TuberiaTensorStat: Codable, Sendable, Equatable, Hashable {
  public let shape: [Int]
  /// Canonical MLX dtype string, e.g. `"float16"`, `"float32"`, `"bfloat16"`,
  /// `"int32"`. Always lowercase, never abbreviated.
  public let dtype: String
  public let min: Double
  public let max: Double
  public let mean: Double
  public let std: Double
  public let hasNaN: Bool
  public let hasInf: Bool

  /// Default magnitude threshold for the `numericalAnomaly` side-channel's
  /// `outOfRange` kind. Callers that emit the anomaly inspect
  /// `abs(stat.max) > threshold || abs(stat.min) > threshold`. Hard-coded at
  /// `1e6` per Q2 of the GLASS PIPES execution plan; expose as a `var` so
  /// downstream tooling can override at process scope without recompiling.
  ///
  /// Marked `nonisolated(unsafe)` because Swift 6 strict concurrency forbids
  /// mutable shared globals; in practice this scalar is read on the hot path
  /// and written, at most, once at startup by host tooling. If a future
  /// caller writes it from multiple threads, the visibility race is benign
  /// (any observed value is still a valid threshold).
  public nonisolated(unsafe) static var defaultOutOfRangeThreshold: Double = 1e6

  public init(
    shape: [Int],
    dtype: String,
    min: Double,
    max: Double,
    mean: Double,
    std: Double,
    hasNaN: Bool,
    hasInf: Bool
  ) {
    self.shape = shape
    self.dtype = dtype
    self.min = min
    self.max = max
    self.mean = mean
    self.std = std
    self.hasNaN = hasNaN
    self.hasInf = hasInf
  }

  /// Sample a tensor's statistics in a single Metal command-buffer.
  ///
  /// Performs eight MLX reductions (`min`, `max`, `mean`, `std`,
  /// `isNaN().any()`, `isInf().any()`), `eval()`s the resulting scalar
  /// MLXArrays once as a tuple so they share one graph submission, then reads
  /// the scalars back as `Double` / `Bool`.
  ///
  /// This function is **pure**: it does not emit telemetry, does not consult
  /// any reporter, and has no out-of-band side effects. Anomaly emission
  /// happens at the call site after inspecting the returned `TuberiaTensorStat`
  /// — see Q4 of the GLASS PIPES execution plan.
  ///
  /// **Caller responsibility.** The eight reductions are not free; callers
  /// MUST guard with `if let telemetry { ... }` so this method never runs when
  /// telemetry is off. The protocol intentionally does not use `@autoclosure`
  /// — the guard lives at the emission site.
  public static func sample(_ array: MLXArray) -> TuberiaTensorStat {
    let shape = array.shape
    let dtype = array.dtype
    let dtypeString = canonicalDTypeString(dtype)

    // Casting to float32 once gives us a stable type for the reductions across
    // float16 / bfloat16 / integer inputs. Doing it here also keeps `std` from
    // overflowing on fp16 tensors with large magnitudes (see T5RMSNorm note in
    // v0.6.1 of AGENTS.md — same hazard class).
    let f32 = array.asType(.float32)

    let minArr = f32.min()
    let maxArr = f32.max()
    let meanArr = f32.mean()
    let stdArr = MLX.std(f32)
    let hasNaNArr = MLX.isNaN(f32).any()
    let hasInfArr = MLX.isInf(f32).any()

    // Single command-buffer submission for all six reductions.
    eval(minArr, maxArr, meanArr, stdArr, hasNaNArr, hasInfArr)

    let minValue: Float = minArr.item(Float.self)
    let maxValue: Float = maxArr.item(Float.self)
    let meanValue: Float = meanArr.item(Float.self)
    let stdValue: Float = stdArr.item(Float.self)
    let hasNaN: Bool = hasNaNArr.item(Bool.self)
    let hasInf: Bool = hasInfArr.item(Bool.self)

    return TuberiaTensorStat(
      shape: shape,
      dtype: dtypeString,
      min: Double(minValue),
      max: Double(maxValue),
      mean: Double(meanValue),
      std: Double(stdValue),
      hasNaN: hasNaN,
      hasInf: hasInf
    )
  }

  /// Map an MLX `DType` to its canonical lowercase string. Mirrors the names
  /// used by `numpy` / `torch` / the Vinetas-host adapter's phase suffix.
  private static func canonicalDTypeString(_ dtype: DType) -> String {
    switch dtype {
    case .bool: return "bool"
    case .uint8: return "uint8"
    case .uint16: return "uint16"
    case .uint32: return "uint32"
    case .uint64: return "uint64"
    case .int8: return "int8"
    case .int16: return "int16"
    case .int32: return "int32"
    case .int64: return "int64"
    case .float16: return "float16"
    case .float32: return "float32"
    case .bfloat16: return "bfloat16"
    case .complex64: return "complex64"
    case .float64: return "float64"
    }
  }
}
