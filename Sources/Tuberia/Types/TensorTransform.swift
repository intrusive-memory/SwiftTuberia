import MLX

/// Optional per-tensor transform applied after key remapping, before quantization.
public typealias TensorTransform = @Sendable (String, MLXArray) -> MLXArray
