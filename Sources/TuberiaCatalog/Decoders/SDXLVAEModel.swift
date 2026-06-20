import Foundation
@preconcurrency import MLX
import MLXNN
import Tuberia

// MARK: - ResnetBlock2D

/// Residual block with GroupNorm + SiLU + Conv2d.
///
/// Architecture:
///   x -> norm1 -> silu -> conv1 -> norm2 -> silu -> conv2 -> + residual -> output
///
/// When input and output channels differ, a 1x1 `conv_shortcut` adapts the residual path.
/// GroupNorm uses 32 groups throughout.
///
/// Input/output layout: [B, H, W, C] (NHWC — MLX convention)
final class ResnetBlock2D: MLXNN.Module {
  let norm1: MLXNN.GroupNorm
  let conv1: MLXNN.Conv2d
  let norm2: MLXNN.GroupNorm
  let conv2: MLXNN.Conv2d
  let convShortcut: MLXNN.Conv2d?

  private static let maxGroups = 32

  /// Computes the largest valid group count for GroupNorm such that
  /// `groupCount <= maxGroups` and `channels % groupCount == 0`.
  ///
  /// This handles the case where the channel count is smaller than `maxGroups`
  /// (e.g. the first ResNet block in the mid-block operates on 4 latent channels).
  private static func groupCount(for channels: Int) -> Int {
    var g = min(maxGroups, channels)
    while g > 1 && channels % g != 0 {
      g -= 1
    }
    return g
  }

  init(inChannels: Int, outChannels: Int) {
    self.norm1 = MLXNN.GroupNorm(
      groupCount: Self.groupCount(for: inChannels),
      dimensions: inChannels,
      pytorchCompatible: true
    )
    self.conv1 = MLXNN.Conv2d(
      inputChannels: inChannels,
      outputChannels: outChannels,
      kernelSize: .init(3),
      padding: .init(1)
    )
    self.norm2 = MLXNN.GroupNorm(
      groupCount: Self.groupCount(for: outChannels),
      dimensions: outChannels,
      pytorchCompatible: true
    )
    self.conv2 = MLXNN.Conv2d(
      inputChannels: outChannels,
      outputChannels: outChannels,
      kernelSize: .init(3),
      padding: .init(1)
    )
    if inChannels != outChannels {
      self.convShortcut = MLXNN.Conv2d(
        inputChannels: inChannels,
        outputChannels: outChannels,
        kernelSize: .init(1)
      )
    } else {
      self.convShortcut = nil
    }
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Force evaluation before GroupNorm. Under memory pressure, lazy tensors produced
    // by large graph operations can have ndim=0 (shapeless). eval() ensures x is
    // materialised and its shape is readable before norm1/norm2 operate on it.
    eval(x)
    var h = x
    h = norm1(h)
    h = h * MLX.sigmoid(h)
    h = conv1(h)
    h = norm2(h)
    h = h * MLX.sigmoid(h)
    h = conv2(h)

    let residual: MLXArray
    if let shortcut = convShortcut {
      residual = shortcut(x)
    } else {
      residual = x
    }
    return h + residual
  }
}

// MARK: - AttentionBlock

/// Single-head self-attention block for the VAE mid-block.
///
/// Reshapes spatial dimensions [B, H, W, C] → [B, H*W, C] for attention,
/// applies Q/K/V projections, computes scaled dot-product attention,
/// projects output, then reshapes back to [B, H, W, C].
///
/// Uses GroupNorm(32) before the attention computation.
final class AttentionBlock: MLXNN.Module {
  let groupNorm: MLXNN.GroupNorm
  let query: MLXNN.Linear
  let key: MLXNN.Linear
  let value: MLXNN.Linear
  let projAttn: MLXNN.Linear

  private static let numGroups = 32

  init(channels: Int) {
    self.groupNorm = MLXNN.GroupNorm(
      groupCount: Self.numGroups,
      dimensions: channels,
      pytorchCompatible: true
    )
    self.query = MLXNN.Linear(channels, channels)
    self.key = MLXNN.Linear(channels, channels)
    self.value = MLXNN.Linear(channels, channels)
    self.projAttn = MLXNN.Linear(channels, channels)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Force evaluation before accessing shape dimensions. Under memory pressure,
    // lazy tensors produced by large graph operations can have ndim=0 (shapeless).
    // eval() ensures x is materialised and its shape is readable.
    eval(x)

    let b = x.dim(0)
    let h = x.dim(1)
    let w = x.dim(2)
    let c = x.dim(3)
    let seqLen = h * w

    // Normalize then flatten spatial dims for attention
    var hidden = groupNorm(x)  // [B, H, W, C]
    hidden = hidden.reshaped([b, seqLen, c])  // [B, H*W, C]

    let q = query(hidden)  // [B, H*W, C]
    let k = key(hidden)  // [B, H*W, C]
    let v = value(hidden)  // [B, H*W, C]

    // Use MLXFast.scaledDotProductAttention (flash attention) instead of materialising
    // an explicit [B, H*W, H*W] attention-weight matrix. For a 1024×1024 image the
    // naïve matmul produces a [1, 16384, 16384] tensor (~536 MB fp16) which can cause
    // out-of-memory failures that surface as shapeless-tensor crashes.
    // Reshape to [B, numHeads=1, seqLen, headDim=C] as required by the API.
    let scale = 1.0 / sqrt(Float(c))
    let qReshaped = q.expandedDimensions(axis: 1)  // [B, 1, H*W, C]
    let kReshaped = k.expandedDimensions(axis: 1)  // [B, 1, H*W, C]
    let vReshaped = v.expandedDimensions(axis: 1)  // [B, 1, H*W, C]

    let attnOut = MLXFast.scaledDotProductAttention(
      queries: qReshaped,
      keys: kReshaped,
      values: vReshaped,
      scale: scale,
      mask: nil
    )  // [B, 1, H*W, C]

    // Remove numHeads dimension and project back to spatial layout
    let squeezed = attnOut.reshaped([b, seqLen, c])  // [B, H*W, C]
    var out = projAttn(squeezed)  // [B, H*W, C]
    out = out.reshaped([b, h, w, c])  // [B, H, W, C]

    return x + out
  }
}

// MARK: - Upsample2D

/// 2x nearest-neighbor upsampling followed by a 3x3 Conv2d.
///
/// Forward: conv(nearest_upsample(x, scale=2))
/// Input/output layout: [B, H, W, C] → [B, 2H, 2W, C]
final class Upsample2D: MLXNN.Module {
  let conv: MLXNN.Conv2d

  init(channels: Int) {
    self.conv = MLXNN.Conv2d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: .init(3),
      padding: .init(1)
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Nearest-neighbor 2x upsample using the MLXNN.Upsample module
    // MLXNN.Upsample operates on [B, H, W, C] (NHWC) convention
    let upsampler = MLXNN.Upsample(scaleFactor: 2.0, mode: .nearest)
    let upsampled = upsampler(x)
    return conv(upsampled)
  }
}

// MARK: - VAEMidBlock

/// Mid-block of the SDXL VAE decoder.
///
/// Architecture: ResnetBlock2D → AttentionBlock → ResnetBlock2D
///
/// The first ResnetBlock expands channels (typically 4 → 512).
/// The attention block operates at the bottleneck resolution.
/// The second ResnetBlock maintains channels at 512.
final class VAEMidBlock: MLXNN.Module {
  let resnets: [ResnetBlock2D]
  let attention: AttentionBlock

  init(inChannels: Int, outChannels: Int) {
    self.resnets = [
      ResnetBlock2D(inChannels: inChannels, outChannels: outChannels),
      ResnetBlock2D(inChannels: outChannels, outChannels: outChannels),
    ]
    self.attention = AttentionBlock(channels: outChannels)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // The interleaved eval()s are load-bearing: they materialize each stage so a
    // shapeless (ndim=0) tensor cannot propagate into the next GroupNorm under
    // memory pressure. Do not remove them.
    var h = resnets[0](x)
    eval(h)
    h = attention(h)
    eval(h)
    h = resnets[1](h)
    eval(h)
    return h
  }
}

// MARK: - VAEUpBlock

/// Up-block of the SDXL VAE decoder.
///
/// Architecture: N × ResnetBlock2D (with optional channel change on first block),
/// followed by an optional Upsample2D for spatial resolution doubling.
///
/// - `inChannels`: input channel count (fed to first ResNet block)
/// - `outChannels`: output channel count (used for remaining blocks + upsample)
/// - `numResnetBlocks`: number of ResnetBlock2D layers (typically 3)
/// - `addUpsample`: whether to append a 2x Upsample2D at the end
final class VAEUpBlock: MLXNN.Module {
  let resnets: [ResnetBlock2D]
  let upsample: Upsample2D?

  init(inChannels: Int, outChannels: Int, numResnetBlocks: Int, addUpsample: Bool) {
    var blocks: [ResnetBlock2D] = []
    for i in 0..<numResnetBlocks {
      let blockIn = i == 0 ? inChannels : outChannels
      blocks.append(ResnetBlock2D(inChannels: blockIn, outChannels: outChannels))
    }
    self.resnets = blocks
    self.upsample = addUpsample ? Upsample2D(channels: outChannels) : nil
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // The interleaved eval()s are load-bearing — they keep a shapeless tensor
    // from reaching the next block under memory pressure. Keep them.
    var h = x
    for resnet in resnets {
      h = resnet(h)
      eval(h)
    }
    if let up = upsample {
      eval(h)
      h = up(h)
    }
    return h
  }
}

// MARK: - SDXLVAEDecoderModel

/// Full SDXL VAE decoder model as an MLXNN.Module.
///
/// Transforms latent tensors [B, H/8, W/8, 4] into pixel data [B, H, W, 3].
///
/// Architecture:
/// ```
/// post_quant_conv (4→4, 1x1)
/// → conv_in (4→512, 3x3, padding=1)
/// → mid_block (ResNet(512→512) + Attention(512) + ResNet(512→512))
/// → up_block_0 (512→512, 3 ResNets, upsample 2x)
/// → up_block_1 (512→512, 3 ResNets, upsample 2x)
/// → up_block_2 (512→256, 3 ResNets, upsample 2x)
/// → up_block_3 (256→128, 3 ResNets, no upsample)
/// → conv_norm_out (GroupNorm 32, 128)
/// → SiLU
/// → conv_out (128→3, 3x3, padding=1)
/// → output [B, H, W, 3]
/// ```
final class SDXLVAEDecoderModel: MLXNN.Module {

  // Post-quantization conv (projects latent channels, 1x1)
  let postQuantConv: MLXNN.Conv2d

  // Channel expansion conv: 4→512, 3×3 with padding
  let convIn: MLXNN.Conv2d

  // Mid block: ResNet(512→512) + Attention(512) + ResNet(512→512)
  let midBlock: VAEMidBlock

  // Up blocks: progressive upsampling + channel reduction
  // up_blocks[0]: 512→512, 3 ResNets, upsample 2x
  // up_blocks[1]: 512→512, 3 ResNets, upsample 2x
  // up_blocks[2]: 512→256, 3 ResNets, upsample 2x
  // up_blocks[3]: 256→128, 3 ResNets, no upsample
  let upBlocks: [VAEUpBlock]

  // Final norm + conv
  let convNormOut: MLXNN.GroupNorm
  let convOut: MLXNN.Conv2d

  private static let numGroups = 32
  private static let latentChannels = 4
  private static let midChannels = 512

  override init() {
    self.postQuantConv = MLXNN.Conv2d(
      inputChannels: Self.latentChannels,
      outputChannels: Self.latentChannels,
      kernelSize: .init(1)
    )

    self.convIn = MLXNN.Conv2d(
      inputChannels: Self.latentChannels,
      outputChannels: Self.midChannels,
      kernelSize: .init(3),
      padding: .init(1)
    )

    self.midBlock = VAEMidBlock(
      inChannels: Self.midChannels,
      outChannels: Self.midChannels
    )

    self.upBlocks = [
      // Block 0: 512→512, 3 ResNets, upsample
      VAEUpBlock(
        inChannels: 512,
        outChannels: 512,
        numResnetBlocks: 3,
        addUpsample: true
      ),
      // Block 1: 512→512, 3 ResNets, upsample
      VAEUpBlock(
        inChannels: 512,
        outChannels: 512,
        numResnetBlocks: 3,
        addUpsample: true
      ),
      // Block 2: 512→256, 3 ResNets, upsample
      VAEUpBlock(
        inChannels: 512,
        outChannels: 256,
        numResnetBlocks: 3,
        addUpsample: true
      ),
      // Block 3: 256→128, 3 ResNets, no upsample
      VAEUpBlock(
        inChannels: 256,
        outChannels: 128,
        numResnetBlocks: 3,
        addUpsample: false
      ),
    ]

    self.convNormOut = MLXNN.GroupNorm(
      groupCount: Self.numGroups,
      dimensions: 128,
      pytorchCompatible: true
    )

    self.convOut = MLXNN.Conv2d(
      inputChannels: 128,
      outputChannels: 3,
      kernelSize: .init(3),
      padding: .init(1)
    )
  }

  /// Decode latents to pixel data.
  ///
  /// - Parameter x: Latent tensor [B, H/8, W/8, 4] (already scaled by 1/scalingFactor)
  /// - Returns: Pixel tensor [B, H, W, 3]
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // The interleaved eval()s are load-bearing: they materialize each stage so a
    // shapeless (ndim=0) tensor cannot propagate forward under memory pressure.
    // Keep them; only the diagnostic prints were removed (#45).
    // Post-quantization projection (1x1 conv, 4→4)
    var h = postQuantConv(x)
    eval(h)

    h = convIn(h)
    eval(h)

    h = midBlock(h)
    eval(h)

    // Progressive upsampling through up blocks
    for upBlock in upBlocks {
      h = upBlock(h)
      eval(h)
    }

    // Final normalization and output projection
    h = convNormOut(h)
    h = h * MLX.sigmoid(h)
    h = convOut(h)

    return h
  }

  /// Tiled decode: bounds the activation transient to roughly one tile
  /// regardless of output resolution (#45). The latent is split into spatial
  /// tiles, each decoded through the full VAE with a `latentOverlap` halo, and
  /// the decoded tiles are recombined with **feathered weighted overlap-add**:
  /// each tile contributes a per-pixel weight that tapers toward 0 across its
  /// haloed edges (the edges shared with a neighbor) and is ~1 in its interior,
  /// and the canvas is normalized by the accumulated weight (`acc / wsum`).
  ///
  /// This is the same overlap-and-blend strategy diffusers' `enable_vae_tiling`
  /// and ComfyUI's tiled VAE use. The VAE mid-block self-attention is **global**
  /// over the bottleneck, so each tile attends only over its own extent and can
  /// drift tile-to-tile by a low-frequency amount. A *hard crop* turns that drift
  /// into a visible seam; the feathered blend spreads it across the overlap band
  /// so the transition is gradient-free. The overlap-add is accumulated in
  /// float32 (then cast back to the input dtype) so fp16 accumulation error does
  /// not reintroduce banding.
  ///
  /// - Parameters:
  ///   - x: scaled latent `[B, H, W, C]` (post-scale, already in decode dtype).
  ///   - latentTile: interior tile edge in *latent* pixels (stride = `latentTile`,
  ///     output stride = `latentTile * 8`).
  ///   - latentOverlap: halo in *latent* pixels added on each side before
  ///     decoding; the feather ramp spans this halo so adjacent tiles blend.
  /// - Returns: pixel tensor `[B, H*8, W*8, 3]`.
  ///
  /// - Note: Opt-in via `SDXLVAEDecoderConfiguration.decodeTileLatentSize`. When
  ///   the latent already fits in one tile this returns the single-pass result
  ///   bit-for-bit (no overlap, no blend), so the default (tiling off) path is
  ///   unchanged. `latentOverlap` must be > 0 for adjacent tiles to blend; with
  ///   0 overlap this degrades to a hard-edged stitch.
  func decodeTiled(_ x: MLXArray, latentTile: Int, latentOverlap: Int) -> MLXArray {
    let scale = 8
    let b = x.dim(0)
    let h = x.dim(1)
    let w = x.dim(2)
    let outH = h * scale
    let outW = w * scale

    // If the latent already fits in a single tile, decode in one pass — the
    // tiled and single-pass results are then identical (no seams, no overlap).
    if h <= latentTile && w <= latentTile {
      return self.callAsFunction(x)
    }

    // Weighted overlap-add accumulators (float32 for blend precision).
    var acc = MLXArray.zeros([b, outH, outW, 3], dtype: .float32)
    var wsum = MLXArray.zeros([1, outH, outW, 1], dtype: .float32)

    let overlapPx = latentOverlap * scale

    var ly = 0
    while ly < h {
      var lx = 0
      while lx < w {
        // Haloed latent window, clamped to the latent bounds.
        let y0 = max(ly - latentOverlap, 0)
        let x0 = max(lx - latentOverlap, 0)
        let y1 = min(ly + latentTile + latentOverlap, h)
        let x1 = min(lx + latentTile + latentOverlap, w)

        let tile = x[0..., y0..<y1, x0..<x1, 0...]
        let decoded = self.callAsFunction(tile).asType(.float32)  // [B, th*8, tw*8, 3]
        eval(decoded)

        // Pixel extent of this decoded (haloed) tile on the full canvas.
        let py0 = y0 * scale
        let px0 = x0 * scale
        let py1 = y1 * scale
        let px1 = x1 * scale
        let tileH = py1 - py0
        let tileW = px1 - px0

        // Feather ramp only on sides that actually abut a neighbor tile (an
        // interior halo). Canvas-boundary sides get full weight so edge pixels
        // are not under-counted. `y0 > 0` ⇒ there is a tile above, etc.
        let mY = Self.featherWeights(
          length: tileH,
          rampHead: y0 > 0 ? overlapPx : 0,
          rampTail: y1 < h ? overlapPx : 0)
        let mX = Self.featherWeights(
          length: tileW,
          rampHead: x0 > 0 ? overlapPx : 0,
          rampTail: x1 < w ? overlapPx : 0)
        // Separable mask [1, tileH, tileW, 1] — broadcasts over batch + channels.
        let mask = (mY.reshaped([tileH, 1]) * mX.reshaped([1, tileW]))
          .reshaped([1, tileH, tileW, 1])

        acc[0..., py0..<py1, px0..<px1, 0...] =
          acc[0..., py0..<py1, px0..<px1, 0...] + decoded * mask
        wsum[0..., py0..<py1, px0..<px1, 0...] =
          wsum[0..., py0..<py1, px0..<px1, 0...] + mask
        eval(acc, wsum)

        lx += latentTile
      }
      ly += latentTile
    }

    // Normalize the weighted sum. wsum is strictly > 0 everywhere (every pixel is
    // covered by at least one tile whose interior weight is 1); clamp guards float
    // underflow in the ramp tails.
    let result = (acc / MLX.maximum(wsum, MLXArray(Float(1e-6)))).asType(x.dtype)
    eval(result)
    return result
  }

  /// Builds a 1-D feather weight vector of `length` pixels: a linear ramp
  /// `0 → 1` over the first `rampHead` pixels, `1 → 0` over the last `rampTail`
  /// pixels, and `1` in between. A ramp width of `0` means that side is flat
  /// (full weight) — used for canvas-boundary edges that have no neighbor to
  /// blend with. Ramps are capped at half the length so head/tail never cross.
  static func featherWeights(length: Int, rampHead: Int, rampTail: Int) -> MLXArray {
    var w = [Float](repeating: 1.0, count: max(length, 0))
    if length <= 0 { return MLXArray(w) }

    let head = min(rampHead, length / 2)
    let tail = min(rampTail, length / 2)

    if head > 0 {
      for i in 0..<head {
        w[i] = Float(i + 1) / Float(head + 1)
      }
    }
    if tail > 0 {
      for i in 0..<tail {
        let v = Float(i + 1) / Float(tail + 1)
        let idx = length - 1 - i
        w[idx] = Swift.min(w[idx], v)
      }
    }
    return MLXArray(w)
  }
}
