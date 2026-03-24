@preconcurrency import MLX

/// Remapped, quantized parameter tensors ready for module assignment.
public struct ModuleParameters: Sendable {
    public let parameters: [String: MLXArray]

    public init(parameters: [String: MLXArray]) {
        self.parameters = parameters
    }
}
