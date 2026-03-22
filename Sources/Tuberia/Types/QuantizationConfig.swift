/// Quantization strategy for weight loading.
public enum QuantizationConfig: Sendable {
    case asStored
    case float16
    case bfloat16
    case int4(groupSize: Int = 64)
    case int8(groupSize: Int = 64)
}
