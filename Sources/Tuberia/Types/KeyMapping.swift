import MLX

/// Key remapping function: safetensors key -> module key. Return nil to skip a key.
public typealias KeyMapping = @Sendable (String) -> String?
