/// Quantization strategy for weight loading.
public enum QuantizationConfig: Sendable {
  case asStored
  case float16
  case bfloat16
  case int4(groupSize: Int = 64)
  case int8(groupSize: Int = 64)
}

extension QuantizationConfig: CustomStringConvertible {
  /// A stable, telemetry-safe string representation of the quantization config.
  ///
  /// Examples: `"asStored"`, `"fp16"`, `"bf16"`, `"int4(64)"`, `"int8(64)"`.
  public var description: String {
    switch self {
    case .asStored:
      return "asStored"
    case .float16:
      return "fp16"
    case .bfloat16:
      return "bf16"
    case .int4(let groupSize):
      return "int4(\(groupSize))"
    case .int8(let groupSize):
      return "int8(\(groupSize))"
    }
  }
}
