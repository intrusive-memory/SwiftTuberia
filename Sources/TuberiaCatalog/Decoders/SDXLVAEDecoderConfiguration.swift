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
  /// Compute dtype for the VAE decode pass. Defaults to **bfloat16** to halve the
  /// decode-time activation footprint vs float32 (#45) while staying numerically
  /// safe: the SDXL VAE is fp16-unstable (`force_upcast=true` in its config) and
  /// an fp16 decode of the real checkpoint overflows to NaN (validated on real
  /// weights in `SDXLVAEDecoderRealWeightsTests`). bfloat16 has the same 16-bit
  /// footprint as fp16 but fp32-class dynamic range, so it captures the memory
  /// win without the overflow. Set to `.float32` to restore the original
  /// full-precision decode; `.float16` is **not** safe for this checkpoint.
  public let decodeDType: DType
  /// Tile edge in *latent* pixels for tiled decode (#45). When non-nil and the
  /// latent is larger than the tile in either spatial dimension, the decode is
  /// performed tile-by-tile to bound the activation transient independently of
  /// output resolution. `nil` (the default) preserves the original single-pass,
  /// behavior-identical decode — callers opt in for high-resolution output where
  /// the single-pass transient would be too large (e.g. 4K).
  ///
  /// Tiles are recombined with a feathered weighted overlap-add (see
  /// `SDXLVAEDecoderModel.decodeTiled`), which keeps the tile boundaries
  /// seam-free even though the VAE mid-block attention is global — validated on
  /// real weights in `SDXLVAEDecoderRealWeightsTests` (tiled-vs-full PSNR > 30 dB,
  /// boundary-gradient ratio ≈ 1.2). It is left off by default because the
  /// single-pass decode is marginally higher-fidelity and cheaper at resolutions
  /// that already fit comfortably in memory.
  public let decodeTileLatentSize: Int?
  /// Halo (in *latent* pixels) added on each side of a tile before decoding; the
  /// feather ramp spans this halo so adjacent tiles blend smoothly. Must be > 0
  /// for the blend to work. Ignored when `decodeTileLatentSize` is nil.
  public let decodeTileLatentOverlap: Int

  public init(
    componentId: String = "sdxl-vae-decoder-fp16",
    latentChannels: Int = 4,
    scalingFactor: Float = 0.13025,
    decodeDType: DType = .bfloat16,
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
