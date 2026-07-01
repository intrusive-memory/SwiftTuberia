@preconcurrency import MLX
import MLXNN
import MLXRandom
import Testing
import Tuberia

@testable import TuberiaCatalog

@Suite("T5XXLEncoder Shape Contract Tests")
struct T5XXLEncoderTests {

  // MARK: - Output Shape Contracts

  @Test("encode produces [1, seq, embeddingDim] output shape")
  func encodeOutputShape() throws {
    let config = T5XXLEncoderConfiguration(maxSequenceLength: 120, embeddingDim: 4096)
    let encoder = try T5XXLEncoder(configuration: config)

    let input = TextEncoderInput(text: "a photo of a cat", maxLength: 120)
    let output = try encoder.encode(input)

    #expect(output.embeddings.shape == [1, 120, 4096])
  }

  @Test("encode produces mask shape [1, seq]")
  func encodeMaskShape() throws {
    let config = T5XXLEncoderConfiguration(maxSequenceLength: 120)
    let encoder = try T5XXLEncoder(configuration: config)

    let input = TextEncoderInput(text: "hello", maxLength: 120)
    let output = try encoder.encode(input)

    #expect(output.mask.shape == [1, 120])
  }

  @Test("Mask has 1s for real tokens and 0s for padding")
  func maskValues() throws {
    let config = T5XXLEncoderConfiguration(maxSequenceLength: 120)
    let encoder = try T5XXLEncoder(configuration: config)

    let input = TextEncoderInput(text: "hi", maxLength: 120)
    let output = try encoder.encode(input)

    eval(output.mask)
    let maskSum = output.mask.sum().item(Float.self)

    #expect(maskSum >= 1.0, "Mask should have at least 1 real token")
    #expect(maskSum < 120.0, "Short text should not fill entire sequence")
  }

  @Test("maxLength clamps to configured maxSequenceLength")
  func respectsMaxLength() throws {
    let config = T5XXLEncoderConfiguration(maxSequenceLength: 50)
    let encoder = try T5XXLEncoder(configuration: config)

    let input = TextEncoderInput(text: "test text", maxLength: 200)
    let output = try encoder.encode(input)

    #expect(output.embeddings.shape[1] == 50)
  }

  @Test("Empty text produces valid output shapes")
  func emptyText() throws {
    let config = T5XXLEncoderConfiguration(maxSequenceLength: 10)
    let encoder = try T5XXLEncoder(configuration: config)

    let input = TextEncoderInput(text: "", maxLength: 10)
    let output = try encoder.encode(input)

    #expect(output.embeddings.shape == [1, 10, 4096])
    #expect(output.mask.shape == [1, 10])
  }

  // MARK: - Weight Lifecycle

  @Test("apply(weights:) loads, unload() clears")
  func weightLifecycle() throws {
    let config = T5XXLEncoderConfiguration()
    let encoder = try T5XXLEncoder(configuration: config)

    #expect(!encoder.isLoaded)

    let weights = Tuberia.ModuleParameters(parameters: ["shared.weight": MLXArray.zeros([4, 4])])
    try encoder.apply(weights: weights)
    #expect(encoder.isLoaded)

    encoder.unload()
    #expect(!encoder.isLoaded)
  }

  @Test("estimatedMemoryBytes is in expected range (~1.2 GB)")
  func estimatedMemory() throws {
    let config = T5XXLEncoderConfiguration()
    let encoder = try T5XXLEncoder(configuration: config)

    let memoryBytes = encoder.estimatedMemoryBytes
    #expect(memoryBytes > 1_000_000_000, "Should be > 1 GB")
    #expect(memoryBytes < 2_000_000_000, "Should be < 2 GB")
  }

  // MARK: - Key Mapping

  @Test("keyMapping keeps encoder keys and skips decoder keys")
  func keyMappingBehavior() throws {
    let config = T5XXLEncoderConfiguration()
    let encoder = try T5XXLEncoder(configuration: config)

    let mapping = encoder.keyMapping

    #expect(mapping("encoder.block.0.layer.0.SelfAttention.q.weight") != nil)
    #expect(mapping("shared.weight") != nil)
    #expect(mapping("decoder.block.0.layer.0.SelfAttention.q.weight") == nil)
    #expect(mapping("lm_head.weight") == nil)
  }

  @Test("int4 sidecar keys map to sibling *_scales / *_biases parameter paths")
  func keyMappingSidecars() throws {
    // The int4-in-memory path requires scales/biases to land on the module's
    // sibling parameters so update(parameters:) can populate them directly.
    #expect(
      T5XXLEncoder.mapKey("encoder.block.0.layer.0.SelfAttention.q.scales")
        == "blocks.0.attention.q_scales")
    #expect(
      T5XXLEncoder.mapKey("encoder.block.0.layer.0.SelfAttention.q.biases")
        == "blocks.0.attention.q_biases")
    #expect(
      T5XXLEncoder.mapKey("encoder.block.3.layer.1.DenseReluDense.wi_0.scales")
        == "blocks.3.ffn.wi_0_scales")
    #expect(
      T5XXLEncoder.mapKey("encoder.block.3.layer.1.DenseReluDense.wo.biases")
        == "blocks.3.ffn.wo_biases")
  }
}

// MARK: - int4-in-memory Parity + Memory

/// Verifies the packed-int4 forward path (`quantizedMM`) is numerically
/// equivalent to the previous dequantize-to-fp16 + plain-matmul path, and that
/// the packed weights are ~4× smaller than their fp16 dequantized form.
///
/// GPU (Metal / MLX) suite — must run on Apple Silicon.
@Suite("T5XXLEncoder int4-in-memory Parity")
struct T5XXLInt4ParityTests {

  // Small self-contained architecture (all dims are multiples of 32 and the
  // contracted dim is a multiple of groupSize=64, as MLX quantize requires).
  private static let hiddenDim = 64
  private static let numHeads = 2
  private static let headDim = 32
  private static let ffnDim = 128
  private static let numBuckets = 32
  private static let numLayers = 2
  private static let vocabSize = 96
  private static let groupSize = 64
  private static let bits = 4

  /// A quantized weight plus the fp16 tensor the old code path would have stored.
  private struct QW {
    let packed: MLXArray  // uint32 [outDim, inDim/8]
    let scales: MLXArray
    let biases: MLXArray
    let dequantTransposed: MLXArray  // fp16 [inDim, outDim] — old matmul layout
  }

  /// Build a quantized weight from a random fp16 reference in PyTorch
  /// `[outDim, inDim]` layout (the safetensors layout).
  private static func makeQW(outDim: Int, inDim: Int, seed: UInt64) -> QW {
    MLXRandom.seed(seed)
    // Small magnitude: T5 attention is intentionally unscaled, so untrained
    // random weights easily overflow fp16 in the QK^T product. Keep weights
    // tiny so the shared forward path stays finite — parity (quant vs dequant)
    // holds at any scale since both paths use the same quantized weights.
    let ref = MLXRandom.normal([outDim, inDim]).asType(.float16) * Float(0.02)
    let (wq, scales, biasesOpt) = quantized(ref, groupSize: groupSize, bits: bits)
    let biases = biasesOpt!
    // Old path: dequantized [outDim, inDim] fp16, transposed to [inDim, outDim].
    let deq = dequantized(wq, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
      .asType(.float16)
      .transposed(1, 0)
    return QW(packed: wq, scales: scales, biases: biases, dequantTransposed: deq)
  }

  private static func makeEncoder() -> T5TransformerEncoder {
    T5TransformerEncoder(
      vocabSize: vocabSize,
      hiddenDim: hiddenDim,
      numLayers: numLayers,
      numHeads: numHeads,
      headDim: headDim,
      ffnDim: ffnDim,
      numBuckets: numBuckets
    )
  }

  @Test("quantizedMM forward matches dequantized+matmul within tight tolerance")
  func forwardParity() throws {
    let H = Self.hiddenDim
    let F = Self.ffnDim

    // Shared non-quantized parameters (identical in both encoders).
    MLXRandom.seed(1)
    let embed = MLXRandom.normal([Self.vocabSize, H]).asType(.float16) * Float(0.1)
    let relBias = MLXRandom.normal([Self.numHeads, Self.numBuckets]).asType(.float16)

    // Per-projection quantized weights, deterministic per (layer, name).
    func qw(_ layer: Int, _ name: Int, outDim: Int, inDim: Int) -> QW {
      Self.makeQW(outDim: outDim, inDim: inDim, seed: UInt64(1000 + layer * 10 + name))
    }

    let quantEnc = Self.makeEncoder()
    let refEnc = Self.makeEncoder()

    for enc in [quantEnc, refEnc] {
      enc.embedding = MLXNN.Embedding(weight: embed)
      enc.final_norm.weight = MLXArray.ones([H]).asType(.float16)
      enc.relative_position_bias = relBias
    }

    for layer in 0..<Self.numLayers {
      // Attention projections: [outDim=H, inDim=H].
      let qP = qw(layer, 0, outDim: H, inDim: H)
      let kP = qw(layer, 1, outDim: H, inDim: H)
      let vP = qw(layer, 2, outDim: H, inDim: H)
      let oP = qw(layer, 3, outDim: H, inDim: H)
      // FFN: wi_* are [outDim=F, inDim=H]; wo is [outDim=H, inDim=F].
      let wi0 = qw(layer, 4, outDim: F, inDim: H)
      let wi1 = qw(layer, 5, outDim: F, inDim: H)
      let woP = qw(layer, 6, outDim: H, inDim: F)

      // Quantized encoder: packed uint32 + sidecars.
      let qa = quantEnc.blocks[layer].attention
      qa.q = qP.packed
      qa.q_scales = qP.scales
      qa.q_biases = qP.biases
      qa.k = kP.packed
      qa.k_scales = kP.scales
      qa.k_biases = kP.biases
      qa.v = vP.packed
      qa.v_scales = vP.scales
      qa.v_biases = vP.biases
      qa.o = oP.packed
      qa.o_scales = oP.scales
      qa.o_biases = oP.biases
      let qf = quantEnc.blocks[layer].ffn
      qf.wi_0 = wi0.packed
      qf.wi_0_scales = wi0.scales
      qf.wi_0_biases = wi0.biases
      qf.wi_1 = wi1.packed
      qf.wi_1_scales = wi1.scales
      qf.wi_1_biases = wi1.biases
      qf.wo = woP.packed
      qf.wo_scales = woP.scales
      qf.wo_biases = woP.biases

      // Reference encoder: dequantized fp16 in [inDim, outDim] layout (old path).
      let ra = refEnc.blocks[layer].attention
      ra.q = qP.dequantTransposed
      ra.k = kP.dequantTransposed
      ra.v = vP.dequantTransposed
      ra.o = oP.dequantTransposed
      let rf = refEnc.blocks[layer].ffn
      rf.wi_0 = wi0.dequantTransposed
      rf.wi_1 = wi1.dequantTransposed
      rf.wo = woP.dequantTransposed
    }

    // Identical input for both. NOTE: the mask is fp32 (as produced by encode()).
    // A fp16 mask would make `(1 - mask) * -1e9` evaluate to `0 * -inf = NaN`
    // inside T5Attention, since -1e9 overflows fp16.
    let seqLen = 8
    let tokens = MLXArray((0..<Int32(seqLen)).map { $0 % Int32(Self.vocabSize) }, [1, seqLen])
    let mask = MLXArray.ones([1, seqLen])

    let outQuant = quantEnc(tokens, attentionMask: mask)
    let outRef = refEnc(tokens, attentionMask: mask)
    eval(outQuant, outRef)

    #expect(outQuant.shape == outRef.shape)

    let diff = (outQuant.asType(.float32) - outRef.asType(.float32))
    let maxAbs = diff.abs().max().item(Float.self)
    let refMag = outRef.asType(.float32).abs().max().item(Float.self)

    // No NaNs / Infs.
    #expect(isNaN(outQuant.asType(.float32)).sum().item(Float.self) == 0)
    #expect(!maxAbs.isNaN && maxAbs.isFinite)

    // quantizedMM is mathematically the same operation as dequantized(...) @ x;
    // the only divergence is fp accumulation order. Expect near-exact parity.
    let tol = max(1e-2, refMag * 5e-3)
    #expect(
      maxAbs <= tol,
      "quantized forward diverged from dequantized reference: maxAbs=\(maxAbs) tol=\(tol) refMag=\(refMag)"
    )
  }

  @Test("packed int4 weights are ~4x smaller than fp16 dequantized")
  func memoryFootprint() throws {
    let H = Self.hiddenDim
    let qw = Self.makeQW(outDim: H, inDim: H, seed: 42)
    eval(qw.packed, qw.scales, qw.biases, qw.dequantTransposed)

    let packedBytes = qw.packed.nbytes + qw.scales.nbytes + qw.biases.nbytes
    let fp16Bytes = qw.dequantTransposed.nbytes

    #expect(qw.packed.dtype == .uint32, "weight must remain packed uint32 in memory")
    // int4 (+ per-group fp16 scales/biases) should be well under half the fp16 size.
    #expect(
      packedBytes < fp16Bytes / 2,
      "expected int4 packed (\(packedBytes) B) << fp16 (\(fp16Bytes) B)"
    )
  }
}
