/// Configuration for a LoRA (Low-Rank Adaptation) adapter.
///
/// Accepts either an Acervo `componentId` (preferred) or a `localPath` (fallback
/// for unregistered adapters). If both are provided, `componentId` takes precedence
/// and `localPath` is ignored. At least one must be non-nil.
public struct LoRAConfig: Sendable {
  /// Acervo component ID for the LoRA adapter safetensors.
  /// Takes precedence over `localPath` if both are provided.
  public let componentId: String?

  /// Local file path -- fallback for adapters not registered in Acervo.
  /// Ignored if `componentId` is non-nil.
  public let localPath: String?

  /// Adapter scale (0.0 = no effect, 1.0 = full effect).
  public let scale: Float

  /// Optional activation keyword to prepend to the prompt.
  public let activationKeyword: String?

  /// At least one of `componentId` or `localPath` must be non-nil.
  /// If both are provided, `componentId` takes precedence and `localPath` is ignored.
  public init(
    componentId: String? = nil, localPath: String? = nil,
    scale: Float = 1.0, activationKeyword: String? = nil
  ) {
    precondition(
      componentId != nil || localPath != nil,
      "LoRAConfig requires at least one of componentId or localPath"
    )
    self.componentId = componentId
    self.localPath = localPath
    self.scale = scale
    self.activationKeyword = activationKeyword
  }
}
