import Tuberia

// MARK: - SDXLVAEDecoder Configuration

public struct SDXLVAEDecoderConfiguration: Sendable {
  /// Acervo component ID for VAE weights.
  public let componentId: String
  /// Number of latent channels the decoder expects.
  public let latentChannels: Int
  /// VAE latent scaling factor (applied internally by the decoder).
  public let scalingFactor: Float

  public init(
    componentId: String = "sdxl-vae-decoder-fp16",
    latentChannels: Int = 4,
    scalingFactor: Float = 0.13025
  ) {
    self.componentId = componentId
    self.latentChannels = latentChannels
    self.scalingFactor = scalingFactor
  }
}
