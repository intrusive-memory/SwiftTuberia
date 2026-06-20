import MLX
import Tuberia

// MARK: - SDXLVAEDecoder Configuration

public struct SDXLVAEDecoderConfiguration: Sendable {
  /// Acervo component ID for VAE weights.
  public let componentId: String
  /// Number of latent channels the decoder expects.
  public let latentChannels: Int
  /// VAE latent scaling factor (applied internally by the decoder).
  public let scalingFactor: Float
  /// Compute dtype for the VAE decode pass. Defaults to fp16 to halve the
  /// decode-time activation footprint (#45). The VAE weights ship as fp16 and
  /// the output is clipped to 8-bit downstream, so a float32 decode wastes ~2×
  /// the activation memory for precision that is discarded anyway. Set to
  /// `.float32` to restore the original full-precision decode if a quality
  /// regression is observed, or `.bfloat16` if an intermediate overflows fp16.
  public let decodeDType: DType
  /// Tile edge in *latent* pixels for tiled decode (#45). When non-nil and the
  /// latent is larger than the tile in either spatial dimension, the decode is
  /// performed tile-by-tile to bound the activation transient independently of
  /// output resolution. `nil` (the default) preserves the original
  /// single-pass, behavior-identical decode.
  ///
  /// - Important: Tiled decode is currently **experimental and off by default**.
  ///   The SDXL VAE mid-block self-attention is global over the bottleneck, so
  ///   each tile attends only over its own spatial extent — small tiles can
  ///   shift low-frequency tone tile-to-tile and leave visible seams. See
  ///   `SDXLVAEDecoderModel.decodeTiled`.
  public let decodeTileLatentSize: Int?
  /// Halo (in *latent* pixels) added on each side of a tile before decoding and
  /// cropped back off after, used to suppress seam artifacts in tiled decode.
  /// Ignored when `decodeTileLatentSize` is nil.
  public let decodeTileLatentOverlap: Int

  public init(
    componentId: String = "sdxl-vae-decoder-fp16",
    latentChannels: Int = 4,
    scalingFactor: Float = 0.13025,
    decodeDType: DType = .float16,
    decodeTileLatentSize: Int? = nil,
    decodeTileLatentOverlap: Int = 8
  ) {
    self.componentId = componentId
    self.latentChannels = latentChannels
    self.scalingFactor = scalingFactor
    self.decodeDType = decodeDType
    self.decodeTileLatentSize = decodeTileLatentSize
    self.decodeTileLatentOverlap = decodeTileLatentOverlap
  }
}
