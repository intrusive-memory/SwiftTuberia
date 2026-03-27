import Foundation
@preconcurrency import MLX
import MLXNN
import Tuberia

/// T5-XXL text encoder producing 4096-dimensional embeddings.
///
/// Conforms to `TextEncoder` + `WeightedSegment`.
///
/// The T5-XXL architecture uses:
/// - Relative position bias (no absolute position embeddings)
/// - RMS layer norm
/// - Gated feed-forward network (GeGLU activation)
/// - 24 transformer encoder layers
/// - Hidden dim: 4096, FFN dim: 10240, heads: 64, head dim: 64
///
/// Key mapping translates T5 safetensors keys (e.g.
/// `encoder.block.{i}.layer.0.SelfAttention.q.weight`) to module
/// property paths (e.g. `blocks.{i}.attention.q`) that match
/// `T5TransformerEncoder`'s property hierarchy.
///
/// Tokenizer: Bundled with weights in the same Acervo component. Loaded via
/// `swift-transformers` `AutoTokenizer.from(modelFolder:)`.
public final class T5XXLEncoder: TextEncoder, @unchecked Sendable {
  public typealias Configuration = T5XXLEncoderConfiguration

  private let configuration: Configuration
  private var transformer: T5TransformerEncoder?
  public private(set) var isLoaded: Bool = false

  // T5-XXL architecture constants
  private static let numLayers = 24
  private static let numHeads = 64
  private static let headDim = 64
  private static let hiddenDim = 4096
  private static let ffnDim = 10240

  public required init(configuration: Configuration) throws {
    self.configuration = configuration
    // Note: In a full implementation, the tokenizer would be loaded here
    // via `withComponentAccess` + `AutoTokenizer.from(modelFolder:)`.
    // For the stub, tokenization is handled by the placeholder encode().
  }

  /// Internal initializer for testing — accepts a pre-built (possibly small-dimension)
  /// `T5TransformerEncoder` so that unit tests don't need to instantiate the full 18GB model.
  internal init(configuration: Configuration, transformer: T5TransformerEncoder) {
    self.configuration = configuration
    self.transformer = transformer
  }

  // MARK: - TextEncoder Protocol

  public var outputEmbeddingDim: Int {
    configuration.embeddingDim
  }

  public var maxSequenceLength: Int {
    configuration.maxSequenceLength
  }

  public func encode(_ input: TextEncoderInput) throws -> TextEncoderOutput {
    let seqLen = min(input.maxLength, configuration.maxSequenceLength)

    if isLoaded, let model = transformer {
      // With real transformer loaded, run the forward pass using placeholder token IDs
      // until a real tokenizer is wired in (Sortie 3).
      let estimatedTokens = min(input.text.count / 4 + 1, seqLen)
      let padTokenId = Int32(0)
      var tokenIds = [Int32](repeating: padTokenId, count: seqLen)
      // Fill up to estimatedTokens with placeholder non-zero IDs
      for i in 0..<min(estimatedTokens, seqLen) {
        tokenIds[i] = Int32(i + 1)
      }
      let tokenTensor = MLXArray(tokenIds, [1, seqLen])

      var maskValues = [Float](repeating: 0.0, count: seqLen)
      for i in 0..<min(estimatedTokens, seqLen) {
        maskValues[i] = 1.0
      }
      let mask = MLXArray(maskValues).reshaped([1, seqLen])

      let embeddings = model(tokenTensor, attentionMask: mask)
      return TextEncoderOutput(embeddings: embeddings, mask: mask)
    } else {
      // Unloaded: produce deterministic output for shape testing
      return placeholderEncode(text: input.text, seqLen: seqLen)
    }
  }

  // MARK: - WeightedSegment

  public var estimatedMemoryBytes: Int {
    // ~1.2 GB for int4 quantized T5-XXL
    1_288_490_188
  }

  /// Translate a T5 safetensors key to the corresponding module parameter path.
  ///
  /// The T5 checkpoint contains keys for both the encoder and decoder halves.
  /// We only need the encoder parameters. Decoder keys and `lm_head.*` are
  /// filtered out by returning `nil`.
  ///
  /// Mapping table (safetensors key → module path):
  /// ```
  /// shared.weight                                              → embedding.weight
  /// encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight → relative_position_bias
  /// encoder.block.{i}.layer.0.layer_norm.weight               → blocks.{i}.pre_attn_norm.weight
  /// encoder.block.{i}.layer.0.SelfAttention.q.weight          → blocks.{i}.attention.q
  /// encoder.block.{i}.layer.0.SelfAttention.k.weight          → blocks.{i}.attention.k
  /// encoder.block.{i}.layer.0.SelfAttention.v.weight          → blocks.{i}.attention.v
  /// encoder.block.{i}.layer.0.SelfAttention.o.weight          → blocks.{i}.attention.o
  /// encoder.block.{i}.layer.1.layer_norm.weight               → blocks.{i}.pre_ffn_norm.weight
  /// encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight      → blocks.{i}.ffn.wi_0
  /// encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight      → blocks.{i}.ffn.wi_1
  /// encoder.block.{i}.layer.1.DenseReluDense.wo.weight        → blocks.{i}.ffn.wo
  /// encoder.final_layer_norm.weight                            → final_norm.weight
  /// decoder.*                                                  → nil (skip)
  /// lm_head.*                                                  → nil (skip)
  /// ```
  public var keyMapping: KeyMapping {
    { key in
      T5XXLEncoder.mapKey(key)
    }
  }

  /// Static key mapping function, separated for testability.
  static func mapKey(_ key: String) -> String? {
    // Skip decoder and language model head keys — encoder-only model
    if key.hasPrefix("decoder.") || key.hasPrefix("lm_head.") {
      return nil
    }

    // Shared token embedding table
    if key == "shared.weight" {
      return "embedding.weight"
    }

    // Final encoder layer norm
    if key == "encoder.final_layer_norm.weight" {
      return "final_norm.weight"
    }

    // Relative position bias — only lives on block 0 in safetensors,
    // but stored at top level of T5TransformerEncoder as `relative_position_bias`.
    if key == "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" {
      return "relative_position_bias"
    }

    // encoder.block.{i}.layer.0.* — attention sub-keys
    // encoder.block.{i}.layer.1.* — FFN sub-keys
    //
    // Pattern: "encoder.block." prefix followed by block index, then ".layer.{j}." then component
    if key.hasPrefix("encoder.block.") {
      let afterPrefix = String(key.dropFirst("encoder.block.".count))
      // afterPrefix is like "0.layer.0.SelfAttention.q.weight" or "0.layer.1.DenseReluDense.wi_0.weight"
      guard let dotIdx = afterPrefix.firstIndex(of: ".") else { return nil }
      let blockIdxStr = String(afterPrefix[afterPrefix.startIndex..<dotIdx])
      guard let blockIdx = Int(blockIdxStr) else { return nil }
      let afterBlockIdx = String(afterPrefix[afterPrefix.index(after: dotIdx)...])
      // afterBlockIdx is like "layer.0.SelfAttention.q.weight" or "layer.1.DenseReluDense.wi_0.weight"

      let prefix = "blocks.\(blockIdx)"

      if afterBlockIdx.hasPrefix("layer.0.layer_norm.weight") {
        return "\(prefix).pre_attn_norm.weight"
      }
      if afterBlockIdx.hasPrefix("layer.0.SelfAttention.q.weight") {
        return "\(prefix).attention.q"
      }
      if afterBlockIdx.hasPrefix("layer.0.SelfAttention.k.weight") {
        return "\(prefix).attention.k"
      }
      if afterBlockIdx.hasPrefix("layer.0.SelfAttention.v.weight") {
        return "\(prefix).attention.v"
      }
      if afterBlockIdx.hasPrefix("layer.0.SelfAttention.o.weight") {
        return "\(prefix).attention.o"
      }
      if afterBlockIdx.hasPrefix("layer.1.layer_norm.weight") {
        return "\(prefix).pre_ffn_norm.weight"
      }
      if afterBlockIdx.hasPrefix("layer.1.DenseReluDense.wi_0.weight") {
        return "\(prefix).ffn.wi_0"
      }
      if afterBlockIdx.hasPrefix("layer.1.DenseReluDense.wi_1.weight") {
        return "\(prefix).ffn.wi_1"
      }
      if afterBlockIdx.hasPrefix("layer.1.DenseReluDense.wo.weight") {
        return "\(prefix).ffn.wo"
      }
      // Unknown sub-key within encoder.block — skip
      return nil
    }

    // Unknown key — skip
    return nil
  }

  public func apply(weights: Tuberia.ModuleParameters) throws {
    // Build the flat key → MLXArray mapping using our key mapping function,
    // then unflatten it into a NestedDictionary<String, MLXArray> that
    // MLXNN.Module.update(parameters:) can consume.
    var flatMapped: [(String, MLXArray)] = []
    for (safetensorsKey, tensor) in weights.parameters {
      if let mappedKey = T5XXLEncoder.mapKey(safetensorsKey) {
        flatMapped.append((mappedKey, tensor))
      }
    }

    // Build or reuse the transformer module
    let model: T5TransformerEncoder
    if let existing = self.transformer {
      model = existing
    } else {
      model = T5TransformerEncoder()
    }

    // Unflatten the mapped parameters into MLXNN.ModuleParameters
    // (NestedDictionary<String, MLXArray>) and load into the module.
    let mlxParams = MLXNN.ModuleParameters.unflattened(flatMapped)
    model.update(parameters: mlxParams)

    self.transformer = model
    self.isLoaded = true
  }

  public func unload() {
    self.transformer = nil
    self.isLoaded = false
  }

  // MARK: - Private

  /// Placeholder encode that produces correctly shaped output.
  /// In the full implementation, this would run the T5 transformer encoder.
  private func placeholderEncode(text: String, seqLen: Int) -> TextEncoderOutput {
    // Simulate tokenization: produce a sequence of `seqLen` token positions.
    // Real tokens would come from the tokenizer; here we use the text length
    // to determine how many tokens are "real" vs padding.
    let estimatedTokens = min(text.count / 4 + 1, seqLen)  // rough char-to-token ratio
    let actualSeqLen = seqLen

    // Embeddings: [1, seqLen, embeddingDim] -- placeholder values
    let embeddings = MLXArray.zeros([1, actualSeqLen, configuration.embeddingDim])

    // Mask: [1, seqLen] -- 1 for real tokens, 0 for padding
    var maskValues = [Float](repeating: 0.0, count: actualSeqLen)
    for i in 0..<min(estimatedTokens, actualSeqLen) {
      maskValues[i] = 1.0
    }
    let mask = MLXArray(maskValues).reshaped([1, actualSeqLen])

    return TextEncoderOutput(embeddings: embeddings, mask: mask)
  }
}
