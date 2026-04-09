import Foundation
@preconcurrency import MLX
import MLXNN
import Tuberia

// MARK: - T5RMSNorm

/// RMS layer normalization as used in T5.
///
/// Unlike LayerNorm, T5RMSNorm has no bias and no mean subtraction.
/// Forward: `x * rsqrt(mean(x^2) + eps) * weight`
public final class T5RMSNorm: MLXNN.Module {
  /// Scale parameter, shape [hidden_dim].
  var weight: MLXArray

  private let eps: Float

  public init(dim: Int, eps: Float = 1e-6) {
    self.weight = MLXArray.ones([dim])
    self.eps = eps
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Compute RMS: mean(x^2) + eps
    let meanSquare = x.square().mean(axis: -1, keepDims: true)
    let rms = MLX.rsqrt(meanSquare + eps)
    return x * rms * weight
  }
}

// MARK: - T5Attention

/// Multi-head self-attention with relative position bias.
///
/// T5 uses:
/// - 64 heads, head_dim 64 (total hidden_dim 4096)
/// - No bias on Q/K/V/O projections
/// - Relative position bias (32 buckets, bidirectional) shared across all layers
///
/// The `relative_attention_bias` embedding is only stored on layer 0's T5Attention
/// and passed in at call time via the `positionBias` parameter.
public final class T5Attention: MLXNN.Module {
  // Projection matrices [hidden_dim, hidden_dim], no bias
  var q: MLXArray
  var k: MLXArray
  var v: MLXArray
  var o: MLXArray

  private let numHeads: Int
  private let headDim: Int
  private let hiddenDim: Int
  private let numBuckets: Int

  public init(
    hiddenDim: Int = 4096,
    numHeads: Int = 64,
    headDim: Int = 64,
    numBuckets: Int = 32
  ) {
    self.hiddenDim = hiddenDim
    self.numHeads = numHeads
    self.headDim = headDim
    self.numBuckets = numBuckets

    // Initialize projection weights with zeros (replaced on weight load)
    self.q = MLXArray.zeros([hiddenDim, hiddenDim])
    self.k = MLXArray.zeros([hiddenDim, hiddenDim])
    self.v = MLXArray.zeros([hiddenDim, hiddenDim])
    self.o = MLXArray.zeros([hiddenDim, hiddenDim])
    super.init()
  }

  /// Forward pass.
  ///
  /// - Parameters:
  ///   - x: Input tensor [B, seq_len, hidden_dim].
  ///   - positionBias: Pre-computed relative position bias [1, num_heads, seq_len, seq_len].
  ///     Pass `nil` to skip (bias already included in positionBias computed externally).
  ///   - mask: Optional attention mask [B, 1, 1, seq_len] (0 = masked, 1 = attended).
  /// - Returns: Output tensor [B, seq_len, hidden_dim].
  public func callAsFunction(
    _ x: MLXArray,
    positionBias: MLXArray? = nil,
    mask: MLXArray? = nil
  ) -> MLXArray {
    let shape = x.shape
    let batchSize = shape[0]
    let seqLen = shape[1]

    // Project Q, K, V: matmul with weight matrices
    // x: [B, S, D], weight: [D, D] -> [B, S, D]
    let qProj = MLX.matmul(x, q)
    let kProj = MLX.matmul(x, k)
    let vProj = MLX.matmul(x, v)

    // Reshape to [B, seq_len, num_heads, head_dim] then transpose to [B, num_heads, seq_len, head_dim]
    let qReshaped = qProj.reshaped([batchSize, seqLen, numHeads, headDim])
      .transposed(0, 2, 1, 3)
    let kReshaped = kProj.reshaped([batchSize, seqLen, numHeads, headDim])
      .transposed(0, 2, 1, 3)
    let vReshaped = vProj.reshaped([batchSize, seqLen, numHeads, headDim])
      .transposed(0, 2, 1, 3)

    // Scaled dot-product attention
    // scores: [B, num_heads, seq_len, seq_len]
    let scale = Float(1.0 / sqrt(Double(headDim)))
    var scores = MLX.matmul(qReshaped, kReshaped.transposed(0, 1, 3, 2)) * scale

    // Add relative position bias if provided
    if let bias = positionBias {
      scores = scores + bias
    }

    // Apply attention mask (convert 0/1 mask to additive -inf mask)
    if let mask = mask {
      // mask: [B, 1, 1, seq_len], values 0.0 (masked) or 1.0 (unmasked)
      // Convert to large negative additive bias where mask == 0
      let maskBias = (1.0 - mask) * Float(-1e9)
      scores = scores + maskBias
    }

    // Softmax over last dim
    let attnWeights = MLX.softmax(scores, axis: -1)

    // Attend to values: [B, num_heads, seq_len, head_dim]
    let attnOutput = MLX.matmul(attnWeights, vReshaped)

    // Transpose and reshape back to [B, seq_len, hidden_dim]
    let attnConcat = attnOutput.transposed(0, 2, 1, 3)
      .reshaped([batchSize, seqLen, hiddenDim])

    // Output projection
    return MLX.matmul(attnConcat, o)
  }
}

// MARK: - T5GatedFFN

/// Gated feed-forward network with GeGLU activation.
///
/// Forward: `wo(gelu(wi_0(x)) * wi_1(x))`
///
/// Parameters (no bias):
/// - `wi_0` [hidden_dim, ffn_dim]
/// - `wi_1` [hidden_dim, ffn_dim]
/// - `wo`   [ffn_dim, hidden_dim]
public final class T5GatedFFN: MLXNN.Module {
  var wi_0: MLXArray
  var wi_1: MLXArray
  var wo: MLXArray

  public init(hiddenDim: Int = 4096, ffnDim: Int = 10240) {
    self.wi_0 = MLXArray.zeros([hiddenDim, ffnDim])
    self.wi_1 = MLXArray.zeros([hiddenDim, ffnDim])
    self.wo = MLXArray.zeros([ffnDim, hiddenDim])
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // GeGLU gate: gelu(wi_0(x)) * wi_1(x)
    // MLXNN.gelu uses compile(shapeless: true) which can return a 0D tensor under memory
    // pressure, causing downstream crashes. Use direct gelu_new (tanh approximation) instead.
    let inner = MLX.matmul(x, wi_0)
    let gate =
      inner * 0.5 * (1.0 + MLX.tanh(0.7978845608 * (inner + 0.044715 * inner * inner * inner)))
    let linear = MLX.matmul(x, wi_1)
    let gated = gate * linear
    // Project back to hidden dim
    return MLX.matmul(gated, wo)
  }
}

// MARK: - T5EncoderBlock

/// Single T5 encoder transformer block.
///
/// Structure:
/// ```
/// x -> pre_attn_norm -> attention -> residual(+x)
///   -> pre_ffn_norm  -> ffn       -> residual(+x)
/// ```
public final class T5EncoderBlock: MLXNN.Module {
  var pre_attn_norm: T5RMSNorm
  var attention: T5Attention
  var pre_ffn_norm: T5RMSNorm
  var ffn: T5GatedFFN

  public init(
    hiddenDim: Int = 4096,
    numHeads: Int = 64,
    headDim: Int = 64,
    ffnDim: Int = 10240,
    numBuckets: Int = 32
  ) {
    self.pre_attn_norm = T5RMSNorm(dim: hiddenDim)
    self.attention = T5Attention(
      hiddenDim: hiddenDim,
      numHeads: numHeads,
      headDim: headDim,
      numBuckets: numBuckets
    )
    self.pre_ffn_norm = T5RMSNorm(dim: hiddenDim)
    self.ffn = T5GatedFFN(hiddenDim: hiddenDim, ffnDim: ffnDim)
    super.init()
  }

  /// Forward pass for a single encoder block.
  ///
  /// - Parameters:
  ///   - x: Input tensor [B, seq_len, hidden_dim].
  ///   - positionBias: Relative position bias [1, num_heads, seq_len, seq_len].
  ///   - mask: Optional attention mask [B, 1, 1, seq_len].
  /// - Returns: Output tensor [B, seq_len, hidden_dim].
  public func callAsFunction(
    _ x: MLXArray,
    positionBias: MLXArray? = nil,
    mask: MLXArray? = nil
  ) -> MLXArray {
    // Self-attention with pre-norm and residual
    let normedForAttn = pre_attn_norm(x)
    let attnOut = attention(normedForAttn, positionBias: positionBias, mask: mask)
    let postAttn = x + attnOut

    // FFN with pre-norm and residual
    let normedForFFN = pre_ffn_norm(postAttn)
    let ffnOut = ffn(normedForFFN)
    return postAttn + ffnOut
  }
}

// MARK: - T5TransformerEncoder

/// Full T5-XXL encoder stack.
///
/// Architecture:
/// ```
/// Token IDs [B, seq_len]
///   -> Embedding [B, seq_len, 4096]
///   -> 24 T5EncoderBlocks (with shared relative position bias)
///   -> Final RMSNorm
///   -> Output embeddings [B, seq_len, 4096]
/// ```
///
/// Relative position bias design:
/// - Single `relative_position_bias` embedding [num_heads, num_buckets] lives here on the encoder.
/// - At forward time, we compute the full [1, num_heads, seq_len, seq_len] bias matrix once
///   and pass it into every encoder block.
/// - This matches T5's original design where the bias table lives on layer 0 but is shared.
public final class T5TransformerEncoder: MLXNN.Module {
  // MARK: Default architecture constants (T5-XXL production values)
  static let defaultVocabSize = 32128
  static let defaultHiddenDim = 4096
  static let defaultNumLayers = 24
  static let defaultNumHeads = 64
  static let defaultHeadDim = 64
  static let defaultFFNDim = 10240
  static let defaultNumBuckets = 32
  static let defaultMaxDistance = 128

  // MARK: Instance architecture dimensions
  // Stored as instance properties so small-dimension variants (used in tests)
  // are fully self-contained and do not accidentally use production-sized constants.
  private let instanceNumHeads: Int
  private let instanceNumBuckets: Int
  private let instanceMaxDistance: Int

  // MARK: Parameters

  /// Token embedding table [vocab_size, hidden_dim].
  var embedding: MLXNN.Embedding

  /// 24 transformer encoder blocks.
  var blocks: [T5EncoderBlock]

  /// Final RMS layer norm.
  var final_norm: T5RMSNorm

  /// Relative position bias table [num_heads, num_buckets].
  /// Single instance shared across all layers (only stored/declared here).
  var relative_position_bias: MLXArray

  public init(
    vocabSize: Int = 32128,
    hiddenDim: Int = 4096,
    numLayers: Int = 24,
    numHeads: Int = 64,
    headDim: Int = 64,
    ffnDim: Int = 10240,
    numBuckets: Int = 32,
    maxDistance: Int = 128
  ) {
    self.instanceNumHeads = numHeads
    self.instanceNumBuckets = numBuckets
    self.instanceMaxDistance = maxDistance

    self.embedding = MLXNN.Embedding(embeddingCount: vocabSize, dimensions: hiddenDim)
    self.blocks = (0..<numLayers).map { _ in
      T5EncoderBlock(
        hiddenDim: hiddenDim,
        numHeads: numHeads,
        headDim: headDim,
        ffnDim: ffnDim,
        numBuckets: numBuckets
      )
    }
    self.final_norm = T5RMSNorm(dim: hiddenDim)
    // Relative position bias table; initialized to zeros, loaded from weights
    self.relative_position_bias = MLXArray.zeros([numHeads, numBuckets])
    super.init()
  }

  // MARK: - Forward

  /// Run the T5 encoder forward pass.
  ///
  /// - Parameters:
  ///   - tokenIds: Integer token IDs [B, seq_len].
  ///   - attentionMask: Optional float mask [B, seq_len] where 1 = attend, 0 = ignore.
  /// - Returns: Contextual embeddings [B, seq_len, 4096].
  public func callAsFunction(
    _ tokenIds: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> MLXArray {
    let shape = tokenIds.shape
    let batchSize = shape[0]
    let seqLen = shape[1]

    // Embed token IDs: [B, seq_len, hidden_dim]
    var hidden = embedding(tokenIds)

    // Compute relative position bias once for all layers
    let posBias = computeRelativePositionBias(seqLen: seqLen)
    // posBias: [1, num_heads, seq_len, seq_len]

    // Reshape attention mask for broadcasting in attention: [B, 1, 1, seq_len]
    let attnMask: MLXArray?
    if let mask = attentionMask {
      attnMask = mask.reshaped([batchSize, 1, 1, seqLen])
    } else {
      attnMask = nil
    }

    // Run through all encoder blocks
    for block in blocks {
      hidden = block(hidden, positionBias: posBias, mask: attnMask)
    }

    // Final layer norm
    hidden = final_norm(hidden)

    return hidden
  }

  // MARK: - Relative Position Bias

  /// Compute the relative position bias matrix from the bucket embedding table.
  ///
  /// T5 uses logarithmically-spaced buckets for relative positions.
  /// For bidirectional attention (encoder), positions in both directions are bucketed.
  ///
  /// - Parameter seqLen: Sequence length.
  /// - Returns: Bias tensor [1, num_heads, seq_len, seq_len].
  private func computeRelativePositionBias(seqLen: Int) -> MLXArray {
    let numBuckets = instanceNumBuckets
    let maxDistance = instanceMaxDistance
    let numHeads = instanceNumHeads

    // Build relative positions matrix [seq_len, seq_len]
    // Entry [i, j] = position of key j relative to query i = j - i
    var bucketIndices = [Int32](repeating: 0, count: seqLen * seqLen)

    for queryIdx in 0..<seqLen {
      for keyIdx in 0..<seqLen {
        let relPos = keyIdx - queryIdx
        let bucket = relativeBucket(
          relativePosition: relPos,
          bidirectional: true,
          numBuckets: numBuckets,
          maxDistance: maxDistance
        )
        bucketIndices[queryIdx * seqLen + keyIdx] = Int32(bucket)
      }
    }

    // Create bucket index tensor [seq_len, seq_len]
    let bucketTensor = MLXArray(bucketIndices, [seqLen, seqLen])

    // Gather from relative_position_bias: [num_heads, num_buckets]
    // Result: [num_heads, seq_len, seq_len]
    // We gather along axis=1 (bucket dimension) for each head
    // relative_position_bias: [num_heads, num_buckets]
    // bucketTensor: [seqLen, seqLen] -> indices into axis=1

    // Flatten bucket indices for gathering
    let flatBuckets = bucketTensor.reshaped([seqLen * seqLen])

    // Gather bias values: relative_position_bias[:, bucketIdx] for each head
    // Take slice from relative_position_bias along axis 1
    let gathered = relative_position_bias.T[flatBuckets]  // [seqLen*seqLen, num_heads]

    // Reshape and permute to [num_heads, seq_len, seq_len]
    let biasMatrix = gathered.reshaped([seqLen, seqLen, numHeads])
      .transposed(2, 0, 1)

    // Add batch dimension: [1, num_heads, seq_len, seq_len]
    return biasMatrix.expandedDimensions(axis: 0)
  }

  /// Map a relative position to a bucket index.
  ///
  /// Bidirectional T5 uses half the buckets for each direction.
  /// Positions are bucketed logarithmically past a linear region.
  private func relativeBucket(
    relativePosition: Int,
    bidirectional: Bool,
    numBuckets: Int,
    maxDistance: Int
  ) -> Int {
    var ret = 0
    var relPos = relativePosition
    var n = numBuckets

    if bidirectional {
      n = n / 2
      if relPos > 0 {
        ret += n
      } else {
        relPos = -relPos
      }
    } else {
      relPos = -min(relPos, 0)
    }

    // Half the buckets are for exact positions in [0, maxExact)
    let maxExact = n / 2
    let isSmall = relPos < maxExact

    if isSmall {
      ret += relPos
    } else {
      // Logarithmically spaced buckets for larger positions
      let relPosDbl = Double(relPos)
      let logBucket =
        log(relPosDbl / Double(maxExact))
        / log(Double(maxDistance) / Double(maxExact))
        * Double(n - maxExact)
      let bucket = maxExact + min(Int(logBucket), n - maxExact - 1)
      ret += bucket
    }

    return ret
  }
}
