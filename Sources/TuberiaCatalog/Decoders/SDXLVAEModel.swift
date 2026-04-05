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
    let shape = x.shape  // [B, H, W, C]
    let b = shape[0]
    let h = shape[1]
    let w = shape[2]
    let c = shape[3]
    let seqLen = h * w

    // Normalize then flatten spatial dims for attention
    var hidden = groupNorm(x)  // [B, H, W, C]
    hidden = hidden.reshaped([b, seqLen, c])  // [B, H*W, C]

    let q = query(hidden)  // [B, H*W, C]
    let k = key(hidden)  // [B, H*W, C]
    let v = value(hidden)  // [B, H*W, C]

    // Scaled dot-product attention: softmax(Q K^T / sqrt(C)) V
    let scale = 1.0 / sqrt(Float(c))
    // q: [B, H*W, C], k: [B, H*W, C]
    // attnWeights = q @ k^T: [B, H*W, H*W]
    let kTransposed = k.transposed(0, 2, 1)  // [B, C, H*W]
    var attnWeights = matmul(q, kTransposed) * scale  // [B, H*W, H*W]
    attnWeights = MLX.softmax(attnWeights, axis: -1)

    // attended: [B, H*W, C]
    var out = matmul(attnWeights, v)

    // Project and reshape back to spatial
    out = projAttn(out)  // [B, H*W, C]
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
    var h = resnets[0](x)
    h = attention(h)
    h = resnets[1](h)
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
    var h = x
    for resnet in resnets {
      h = resnet(h)
    }
    if let up = upsample {
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
/// → mid_block (ResNet(4→512) + Attention(512) + ResNet(512→512))
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

  // Mid block: ResNet(4→512) + Attention(512) + ResNet(512→512)
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

    self.midBlock = VAEMidBlock(
      inChannels: Self.latentChannels,
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
    // Post-quantization projection (1x1 conv, 4→4)
    var h = postQuantConv(x)

    // Mid block: bottleneck processing at low resolution
    h = midBlock(h)

    // Progressive upsampling through up blocks
    for upBlock in upBlocks {
      h = upBlock(h)
    }

    // Final normalization and output projection
    h = convNormOut(h)
    h = h * MLX.sigmoid(h)
    h = convOut(h)

    return h
  }
}
