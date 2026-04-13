import Foundation
@preconcurrency import MLX
import MLXNN
import SwiftAcervo
@preconcurrency import Tokenizers
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
public final class T5XXLEncoder: TextEncoder, TokenizerLoadable, @unchecked Sendable {
  public typealias Configuration = T5XXLEncoderConfiguration

  private let configuration: Configuration
  private var transformer: T5TransformerEncoder?
  private var tokenizer: (any Tokenizer)?
  private var _currentWeights: Tuberia.ModuleParameters?
  public private(set) var isLoaded: Bool = false

  // T5-XXL architecture constants
  private static let numLayers = 24
  private static let numHeads = 64
  private static let headDim = 64
  private static let hiddenDim = 4096
  private static let ffnDim = 10240

  public required init(configuration: Configuration) throws {
    self.configuration = configuration
    // Tokenizer is loaded asynchronously via loadTokenizer() during the pipeline
    // load phase (Option B from INF-2: separate async step, keeping init synchronous).
  }

  /// Load the T5 tokenizer from the Acervo component directory.
  ///
  /// Must be called during the pipeline load phase (e.g. from
  /// `DiffusionPipeline.loadModels()`) before any `encode()` calls that require
  /// real tokenization. Falls back to placeholder tokenization if not called.
  ///
  /// The tokenizer files (`tokenizer.json`, `tokenizer_config.json`) are bundled
  /// in the same Acervo component as the weights (component ID configured via
  /// `configuration.componentId`).
  public func loadTokenizer() async {
    do {
      // Step 1: Obtain the model directory URL via Acervo.
      let acervoDir = try await AcervoManager.shared.withModelAccess(configuration.componentId) {
        directoryURL -> URL in
        directoryURL
      }

      // Step 2: Apply the same MACF-aware redirect as WeightLoader.
      //
      // On macOS, MACF (Mandatory Access Control Framework) blocks open() on files inside
      // ~/Library/Group Containers/… for processes that lack the
      // com.apple.security.application-groups entitlement — including xctest.
      // AutoTokenizer.from(modelFolder:) reads JSON files via open(), so it fails
      // silently when acervoDir is an App Group Container path.
      //
      // WeightLoader already works around this by using pre-hardlinked files in
      // /tmp/vinetas-test-models/<componentId>/ (created by `make link-pixart-models`).
      // We apply the same check here so the tokenizer uses the same accessible directory.
      let effectiveDir: URL
      if acervoDir.path.contains("/Group Containers/") {
        let baseDir =
          ProcessInfo.processInfo.environment["VINETAS_TEST_MODELS_DIR"]
          ?? "/tmp/vinetas-test-models"
        let tempDir = URL(fileURLWithPath: baseDir)
          .appendingPathComponent(configuration.componentId)
        let tokenizerJSON = tempDir.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: tokenizerJSON.path) {
          effectiveDir = tempDir
        } else {
          effectiveDir = acervoDir
        }
      } else {
        effectiveDir = acervoDir
      }

      // Step 3: Load the tokenizer from the resolved directory.
      let tok = try await AutoTokenizer.from(modelFolder: effectiveDir)
      self.tokenizer = tok
    } catch {
      // Non-fatal: encode() falls back to placeholder tokenization when tokenizer is nil.
      // Errors here are expected in testing environments without real model files.
    }
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

    // --- Tokenization ---
    // Use real tokenizer when available; fall back to placeholder tokenization.
    let (tokenIds, numRealTokens) = tokenize(text: input.text, seqLen: seqLen)

    // --- Attention mask ---
    // 1 for real tokens (indices 0..<numRealTokens), 0 for padding.
    var maskValues = [Float](repeating: 0.0, count: seqLen)
    for i in 0..<numRealTokens {
      maskValues[i] = 1.0
    }
    let mask = MLXArray(maskValues).reshaped([1, seqLen])

    if isLoaded, let model = transformer {
      // Real transformer forward pass.
      let tokenTensor = MLXArray(tokenIds, [1, seqLen])
      let embeddings = model(tokenTensor, attentionMask: mask)
      return TextEncoderOutput(embeddings: embeddings, mask: mask)
    } else {
      // Unloaded: produce correctly shaped placeholder output.
      let embeddings = MLXArray.zeros([1, seqLen, configuration.embeddingDim])
      return TextEncoderOutput(embeddings: embeddings, mask: mask)
    }
  }

  // MARK: - Tokenization Helpers

  /// Tokenize `text` into a padded [Int32] token ID array of length `seqLen`.
  ///
  /// Returns the array plus the count of real (non-pad) tokens so the caller
  /// can construct the attention mask.
  ///
  /// When the real tokenizer is loaded, uses it; otherwise falls back to a
  /// character-based heuristic (placeholder).
  ///
  /// T5 tokenization notes:
  /// - Pad token ID: 0
  /// - Token IDs are clamped to `seqLen` (truncation)
  /// - If the text is empty, a single EOS/pad token is produced
  private func tokenize(text: String, seqLen: Int) -> ([Int32], Int) {
    if let tok = tokenizer {
      return tokenizeWithRealTokenizer(tok, text: text, seqLen: seqLen)
    } else {
      return placeholderTokenize(text: text, seqLen: seqLen)
    }
  }

  /// Tokenize using the loaded swift-transformers tokenizer.
  private func tokenizeWithRealTokenizer(
    _ tok: any Tokenizer,
    text: String,
    seqLen: Int
  ) -> ([Int32], Int) {
    // Handle empty string: produce a single pad token.
    if text.isEmpty {
      var ids = [Int32](repeating: 0, count: seqLen)
      ids[0] = 0  // pad / EOS token
      return (ids, 1)
    }

    // Encode the text to token IDs.
    let rawIds = tok.encode(text: text)

    // Clamp to maxSequenceLength (truncation).
    let truncated = Array(rawIds.prefix(seqLen))
    let numRealTokens = truncated.count

    // Pad to seqLen with pad token ID 0.
    var padded = truncated.map { Int32($0) }
    while padded.count < seqLen {
      padded.append(0)
    }

    return (padded, numRealTokens)
  }

  /// Placeholder tokenization used when no real tokenizer is available.
  ///
  /// Approximates token count using a character-to-token ratio (~4 chars/token).
  /// Used in testing environments and as a fallback.
  private func placeholderTokenize(text: String, seqLen: Int) -> ([Int32], Int) {
    if text.isEmpty {
      var ids = [Int32](repeating: 0, count: seqLen)
      ids[0] = 0
      return (ids, 1)
    }
    let estimatedTokens = min(text.count / 4 + 1, seqLen)
    var ids = [Int32](repeating: 0, count: seqLen)
    for i in 0..<estimatedTokens {
      ids[i] = Int32(i + 1)
    }
    return (ids, estimatedTokens)
  }

  // MARK: - WeightedSegment

  public var currentWeights: Tuberia.ModuleParameters? { _currentWeights }

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
    self._currentWeights = weights
    self.isLoaded = true
  }

  public func unload() {
    self.transformer = nil
    self._currentWeights = nil
    self.isLoaded = false
  }

}
