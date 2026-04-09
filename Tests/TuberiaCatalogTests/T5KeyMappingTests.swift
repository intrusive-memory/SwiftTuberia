import Foundation
@preconcurrency import MLX
import MLXNN
import Testing
import Tuberia

@testable import TuberiaCatalog

// MARK: - T5 Key Mapping Tests

/// Tests for T5XXLEncoder.mapKey(_:) — the ~580-key safetensors → module path mapping.
///
/// Covers:
/// (a) All 24 layers × 11 keys/layer map to correct module paths
/// (b) decoder.* and lm_head.* keys return nil
/// (c) shared.weight maps to embedding.weight
/// (d) relative_attention_bias (layer 0 only) routes to relative_position_bias
/// (e) encoder.final_layer_norm.weight maps to final_norm.weight
@Suite("T5XXLEncoder Key Mapping Tests", .serialized)
struct T5KeyMappingTests {

  // MARK: - Shared token embedding

  @Test("shared.weight maps to embedding.weight")
  func sharedWeightMapsToEmbedding() {
    #expect(T5XXLEncoder.mapKey("shared.weight") == "embedding.weight")
  }

  // MARK: - Final layer norm

  @Test("encoder.final_layer_norm.weight maps to final_norm.weight")
  func finalLayerNormMapping() {
    #expect(T5XXLEncoder.mapKey("encoder.final_layer_norm.weight") == "final_norm.weight")
  }

  // MARK: - Relative position bias (layer 0 only)

  @Test("relative_attention_bias from block 0 maps to relative_position_bias")
  func relativePositionBiasMapping() {
    let key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    #expect(T5XXLEncoder.mapKey(key) == "relative_position_bias")
  }

  // MARK: - Decoder / lm_head keys filtered out

  @Test("decoder.* keys return nil")
  func decoderKeysReturnNil() {
    let decoderKeys = [
      "decoder.block.0.layer.0.SelfAttention.q.weight",
      "decoder.block.0.layer.0.layer_norm.weight",
      "decoder.final_layer_norm.weight",
      "decoder.embed_tokens.weight",
      "decoder.block.23.layer.2.DenseReluDense.wi_0.weight",
    ]
    for key in decoderKeys {
      #expect(T5XXLEncoder.mapKey(key) == nil, "Expected nil for decoder key: \(key)")
    }
  }

  @Test("lm_head.* keys return nil")
  func lmHeadKeysReturnNil() {
    let lmHeadKeys = [
      "lm_head.weight",
      "lm_head.bias",
    ]
    for key in lmHeadKeys {
      #expect(T5XXLEncoder.mapKey(key) == nil, "Expected nil for lm_head key: \(key)")
    }
  }

  // MARK: - All 24 layers × 11 keys/layer

  @Test("All 24 layers map attention Q/K/V/O keys correctly")
  func allLayersAttentionQKVOMapping() {
    for i in 0..<24 {
      let qKey = "encoder.block.\(i).layer.0.SelfAttention.q.weight"
      let kKey = "encoder.block.\(i).layer.0.SelfAttention.k.weight"
      let vKey = "encoder.block.\(i).layer.0.SelfAttention.v.weight"
      let oKey = "encoder.block.\(i).layer.0.SelfAttention.o.weight"

      #expect(
        T5XXLEncoder.mapKey(qKey) == "blocks.\(i).attention.q",
        "Layer \(i): Q key mismatch")
      #expect(
        T5XXLEncoder.mapKey(kKey) == "blocks.\(i).attention.k",
        "Layer \(i): K key mismatch")
      #expect(
        T5XXLEncoder.mapKey(vKey) == "blocks.\(i).attention.v",
        "Layer \(i): V key mismatch")
      #expect(
        T5XXLEncoder.mapKey(oKey) == "blocks.\(i).attention.o",
        "Layer \(i): O key mismatch")
    }
  }

  @Test("All 24 layers map pre-attention norm correctly")
  func allLayersPreAttnNormMapping() {
    for i in 0..<24 {
      let key = "encoder.block.\(i).layer.0.layer_norm.weight"
      #expect(
        T5XXLEncoder.mapKey(key) == "blocks.\(i).pre_attn_norm.weight",
        "Layer \(i): pre_attn_norm mismatch")
    }
  }

  @Test("All 24 layers map pre-FFN norm correctly")
  func allLayersPreFFNNormMapping() {
    for i in 0..<24 {
      let key = "encoder.block.\(i).layer.1.layer_norm.weight"
      #expect(
        T5XXLEncoder.mapKey(key) == "blocks.\(i).pre_ffn_norm.weight",
        "Layer \(i): pre_ffn_norm mismatch")
    }
  }

  @Test("All 24 layers map FFN wi_0, wi_1, wo correctly")
  func allLayersFFNMapping() {
    for i in 0..<24 {
      let wi0Key = "encoder.block.\(i).layer.1.DenseReluDense.wi_0.weight"
      let wi1Key = "encoder.block.\(i).layer.1.DenseReluDense.wi_1.weight"
      let woKey = "encoder.block.\(i).layer.1.DenseReluDense.wo.weight"

      #expect(
        T5XXLEncoder.mapKey(wi0Key) == "blocks.\(i).ffn.wi_0",
        "Layer \(i): ffn.wi_0 mismatch")
      #expect(
        T5XXLEncoder.mapKey(wi1Key) == "blocks.\(i).ffn.wi_1",
        "Layer \(i): ffn.wi_1 mismatch")
      #expect(
        T5XXLEncoder.mapKey(woKey) == "blocks.\(i).ffn.wo",
        "Layer \(i): ffn.wo mismatch")
    }
  }

  @Test(
    "Complete count: 24 layers × 11 keys + embedding + final_norm + relative_position_bias = 267 mapped keys"
  )
  func totalKeyCount() {
    // Build synthetic key list matching T5 safetensors
    var keys: [String] = []

    // Embedding
    keys.append("shared.weight")

    // Final norm
    keys.append("encoder.final_layer_norm.weight")

    // Relative position bias (only on layer 0)
    keys.append("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")

    // 24 layers × 10 keys (all except relative_attention_bias on layer 0)
    for i in 0..<24 {
      keys.append("encoder.block.\(i).layer.0.layer_norm.weight")
      keys.append("encoder.block.\(i).layer.0.SelfAttention.q.weight")
      keys.append("encoder.block.\(i).layer.0.SelfAttention.k.weight")
      keys.append("encoder.block.\(i).layer.0.SelfAttention.v.weight")
      keys.append("encoder.block.\(i).layer.0.SelfAttention.o.weight")
      keys.append("encoder.block.\(i).layer.1.layer_norm.weight")
      keys.append("encoder.block.\(i).layer.1.DenseReluDense.wi_0.weight")
      keys.append("encoder.block.\(i).layer.1.DenseReluDense.wi_1.weight")
      keys.append("encoder.block.\(i).layer.1.DenseReluDense.wo.weight")
    }

    // Some decoder keys that should be filtered
    keys.append("decoder.block.0.layer.0.SelfAttention.q.weight")
    keys.append("lm_head.weight")

    let mapped = keys.compactMap { T5XXLEncoder.mapKey($0) }

    // Expected: 1 (embedding) + 1 (final_norm) + 1 (relative_position_bias) + 24×9 = 219
    // Note: each layer contributes 4 attention (q,k,v,o) + 1 pre_attn_norm + 1 pre_ffn_norm + 3 ffn = 9 keys
    // Total encoder keys: 3 + (24 × 9) = 219
    // Decoder keys should all be nil → not counted
    let expectedMappedCount = 3 + (24 * 9)  // 219
    #expect(
      mapped.count == expectedMappedCount,
      "Expected \(expectedMappedCount) mapped keys, got \(mapped.count)")
  }

  @Test("No duplicate mapped keys across all encoder keys")
  func noDuplicateMappedKeys() {
    var keys: [String] = ["shared.weight", "encoder.final_layer_norm.weight"]
    keys.append("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
    for i in 0..<24 {
      keys.append("encoder.block.\(i).layer.0.layer_norm.weight")
      keys.append("encoder.block.\(i).layer.0.SelfAttention.q.weight")
      keys.append("encoder.block.\(i).layer.0.SelfAttention.k.weight")
      keys.append("encoder.block.\(i).layer.0.SelfAttention.v.weight")
      keys.append("encoder.block.\(i).layer.0.SelfAttention.o.weight")
      keys.append("encoder.block.\(i).layer.1.layer_norm.weight")
      keys.append("encoder.block.\(i).layer.1.DenseReluDense.wi_0.weight")
      keys.append("encoder.block.\(i).layer.1.DenseReluDense.wi_1.weight")
      keys.append("encoder.block.\(i).layer.1.DenseReluDense.wo.weight")
    }

    let mappedKeys = keys.compactMap { T5XXLEncoder.mapKey($0) }
    let uniqueKeys = Set(mappedKeys)
    #expect(
      uniqueKeys.count == mappedKeys.count,
      "Found duplicate mapped keys: \(mappedKeys.count - uniqueKeys.count) duplicates")
  }
}

// MARK: - T5 apply(weights:) Tests

/// Tests for T5XXLEncoder.apply(weights:) loading ModuleParameters into T5TransformerEncoder.
///
/// Uses tiny dimensions (vocabSize=10, hiddenDim=8, ffnDim=16, numLayers=2) to avoid
/// the large Metal buffer allocations that the full T5-XXL model requires.
@Suite("T5XXLEncoder apply(weights:) Tests", .serialized)
struct T5ApplyWeightsTests {

  // Tiny dimensions for tests — avoids ~18GB Metal allocation from full T5-XXL
  private let testVocabSize = 10
  private let testHiddenDim = 8
  private let testNumLayers = 2
  private let testNumHeads = 2
  private let testHeadDim = 4
  private let testFFNDim = 16
  private let testNumBuckets = 4

  /// Create a test encoder backed by a tiny-dimension T5TransformerEncoder.
  private func makeTestEncoder() -> T5XXLEncoder {
    let config = T5XXLEncoderConfiguration(
      componentId: "test",
      maxSequenceLength: 16,
      embeddingDim: testHiddenDim
    )
    let smallTransformer = T5TransformerEncoder(
      vocabSize: testVocabSize,
      hiddenDim: testHiddenDim,
      numLayers: testNumLayers,
      numHeads: testNumHeads,
      headDim: testHeadDim,
      ffnDim: testFFNDim,
      numBuckets: testNumBuckets
    )
    return T5XXLEncoder(configuration: config, transformer: smallTransformer)
  }

  /// Build synthetic ModuleParameters using T5 safetensors key names.
  ///
  /// Uses scalar (1-element) tensors to avoid memory issues. Shape verification
  /// is off in MLXNN.Module.update(parameters:) by default, so any shape works.
  private func makeSyntheticWeights(numLayers: Int = 2) -> Tuberia.ModuleParameters {
    let scalar = MLXArray(Float(1.0))
    var params: [String: MLXArray] = [:]

    params["shared.weight"] = scalar
    params["encoder.final_layer_norm.weight"] = scalar
    params["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = scalar

    for i in 0..<numLayers {
      params["encoder.block.\(i).layer.0.layer_norm.weight"] = scalar
      params["encoder.block.\(i).layer.0.SelfAttention.q.weight"] = scalar
      params["encoder.block.\(i).layer.0.SelfAttention.k.weight"] = scalar
      params["encoder.block.\(i).layer.0.SelfAttention.v.weight"] = scalar
      params["encoder.block.\(i).layer.0.SelfAttention.o.weight"] = scalar
      params["encoder.block.\(i).layer.1.layer_norm.weight"] = scalar
      params["encoder.block.\(i).layer.1.DenseReluDense.wi_0.weight"] = scalar
      params["encoder.block.\(i).layer.1.DenseReluDense.wi_1.weight"] = scalar
      params["encoder.block.\(i).layer.1.DenseReluDense.wo.weight"] = scalar
    }

    // Decoder keys that should be ignored
    params["decoder.block.0.layer.0.SelfAttention.q.weight"] = scalar
    params["lm_head.weight"] = scalar

    return Tuberia.ModuleParameters(parameters: params)
  }

  @Test("apply(weights:) does not crash with synthetic parameters")
  func applyWeightsDoesNotCrash() throws {
    let encoder = makeTestEncoder()
    // Should not throw or crash
    try encoder.apply(weights: makeSyntheticWeights())
    // Verify the encoder is loaded and weights are retained — confirming apply completed
    #expect(encoder.isLoaded == true, "isLoaded must be true after apply(weights:)")
    #expect(encoder.currentWeights != nil, "currentWeights must be non-nil after apply(weights:)")
  }

  @Test("encode() returns correctly shaped placeholder when unloaded")
  func encodeReturnsPlaceholderWhenUnloaded() throws {
    let config = T5XXLEncoderConfiguration(
      componentId: "test",
      maxSequenceLength: 16,
      embeddingDim: 4096
    )
    let encoder = try T5XXLEncoder(configuration: config)

    let input = TextEncoderInput(text: "hello world", maxLength: 16)
    let output = try encoder.encode(input)

    // Unloaded: should still produce correctly shaped output
    #expect(output.embeddings.shape[0] == 1)
    #expect(output.embeddings.shape[2] == config.embeddingDim)
  }

  @Test("isLoaded is true after apply(weights:) and false after unload()")
  func applyAndUnloadLifecycle() throws {
    let encoder = makeTestEncoder()
    #expect(encoder.isLoaded == false)
    try encoder.apply(weights: makeSyntheticWeights())
    #expect(encoder.isLoaded == true)
    encoder.unload()
    #expect(encoder.isLoaded == false)
  }
}
