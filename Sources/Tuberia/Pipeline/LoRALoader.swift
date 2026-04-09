import Foundation
@preconcurrency import MLX

/// Loads and applies LoRA (Low-Rank Adaptation) adapter weights to model parameters.
///
/// LoRA adapters are stored as safetensors files containing paired `lora_A` and `lora_B`
/// matrices for each adapted layer. The merge formula is:
///
///     W' = W + scale * (B @ A)
///
/// where `W` is the original weight, `A` and `B` are the low-rank adapter matrices,
/// and `scale` is the user-configurable adapter strength.
///
/// The loader reuses the backbone's `keyMapping` to translate LoRA key names to module paths.
/// No explicit target layer declaration is needed -- the LoRA file itself defines which
/// layers are adapted.
///
/// **Constraint**: Single active LoRA per generation (v1).
public struct LoRALoader: Sendable {

  /// Load LoRA adapter weights from their source.
  ///
  /// - Parameters:
  ///   - config: The LoRA configuration specifying the adapter source.
  ///   - keyMapping: The backbone's key mapping to translate LoRA key names.
  /// - Returns: Raw LoRA parameters (lora_A and lora_B pairs, keyed by original names).
  /// - Throws: `PipelineError.weightLoadingFailed` if loading fails.
  public static func loadAdapterWeights(
    config: LoRAConfig,
    keyMapping: KeyMapping
  ) async throws -> ModuleParameters {
    if let componentId = config.componentId {
      return try await WeightLoader.load(
        componentId: componentId,
        keyMapping: keyMapping
      )
    } else if let localPath = config.localPath {
      return try await WeightLoader.loadFromPath(
        localPath,
        keyMapping: keyMapping
      )
    } else {
      // Should not happen due to precondition in LoRAConfig.init
      throw PipelineError.weightLoadingFailed(
        component: "unknown",
        reason: "LoRAConfig has neither componentId nor localPath"
      )
    }
  }

  /// Apply LoRA adapter weights to base model parameters.
  ///
  /// For each layer key `K` in the base parameters, looks for corresponding
  /// `K.lora_A` and `K.lora_B` keys in the adapter weights. If found,
  /// merges them: `W' = W + scale * (B @ A)`.
  ///
  /// - Parameters:
  ///   - adapterWeights: The loaded LoRA parameters containing lora_A/lora_B pairs.
  ///   - baseParameters: The current base model parameters to merge into.
  ///   - scale: Adapter strength (0.0 = no effect, 1.0 = full effect).
  /// - Returns: A new `ModuleParameters` with LoRA merged into matching keys.
  ///   Keys not matched by the adapter are passed through unchanged.
  public static func apply(
    adapterWeights: ModuleParameters,
    to baseParameters: ModuleParameters,
    scale: Float
  ) -> ModuleParameters {
    var merged = baseParameters.parameters

    // Group adapter weights by their base key (strip .lora_A / .lora_B suffix)
    let adapterPairs = groupAdapterPairs(adapterWeights.parameters)

    for (baseKey, pair) in adapterPairs {
      guard let loraA = pair.a, let loraB = pair.b else {
        // Incomplete pair -- skip (a without b or vice versa)
        continue
      }

      guard let baseWeight = merged[baseKey] else {
        // Base key not found in model parameters -- skip
        continue
      }

      // LoRA merge: W' = W + scale * (B @ A)
      let delta = matmul(loraB, loraA)
      let scaled = delta * scale
      merged[baseKey] = baseWeight + scaled
    }

    return ModuleParameters(parameters: merged)
  }

  /// Remove LoRA adaptation by restoring original base weights.
  ///
  /// This reverses the merge operation by subtracting the LoRA delta.
  ///
  /// - Parameters:
  ///   - adapterWeights: The same LoRA parameters that were applied.
  ///   - mergedParameters: The currently merged parameters.
  ///   - scale: The same scale that was used during apply.
  /// - Returns: A new `ModuleParameters` with the LoRA delta removed.
  public static func unapply(
    adapterWeights: ModuleParameters,
    from mergedParameters: ModuleParameters,
    scale: Float
  ) -> ModuleParameters {
    var restored = mergedParameters.parameters

    let adapterPairs = groupAdapterPairs(adapterWeights.parameters)

    for (baseKey, pair) in adapterPairs {
      guard let loraA = pair.a, let loraB = pair.b else {
        continue
      }

      guard let currentWeight = restored[baseKey] else {
        continue
      }

      // Reverse the merge: W = W' - scale * (B @ A)
      let delta = matmul(loraB, loraA)
      let scaled = delta * scale
      restored[baseKey] = currentWeight - scaled
    }

    return ModuleParameters(parameters: restored)
  }

  // MARK: - Private Helpers

  /// A paired set of LoRA adapter matrices for a single layer.
  private struct AdapterPair {
    var a: MLXArray?
    var b: MLXArray?
  }

  /// Group LoRA weights by their base key, extracting lora_A and lora_B pairs.
  ///
  /// LoRA keys follow the convention:
  /// - `some.layer.weight.lora_A` (or `some.layer.lora_A.weight`)
  /// - `some.layer.weight.lora_B` (or `some.layer.lora_B.weight`)
  ///
  /// This function strips the LoRA suffix to find the base key, and groups
  /// the A and B matrices together.
  private static func groupAdapterPairs(
    _ weights: [String: MLXArray]
  ) -> [String: AdapterPair] {
    var pairs: [String: AdapterPair] = [:]

    for (key, tensor) in weights {
      let (baseKey, component) = parseLoRAKey(key)
      guard let component = component else {
        // Not a LoRA key -- skip
        continue
      }

      var pair = pairs[baseKey, default: AdapterPair()]
      switch component {
      case .a:
        pair.a = tensor
      case .b:
        pair.b = tensor
      }
      pairs[baseKey] = pair
    }

    return pairs
  }

  private enum LoRAComponent {
    case a, b
  }

  /// Parse a LoRA key into its base key and component (A or B).
  ///
  /// Handles common conventions:
  /// - `layer.weight.lora_A` -> ("layer.weight", .a)
  /// - `layer.lora_A.weight` -> ("layer.weight", .a)
  /// - `layer.lora_down.weight` -> ("layer.weight", .a)
  /// - `layer.lora_up.weight` -> ("layer.weight", .b)
  private static func parseLoRAKey(_ key: String) -> (baseKey: String, component: LoRAComponent?) {
    // Standard HuggingFace convention: key ends with .lora_A or .lora_B
    if key.hasSuffix(".lora_A") {
      let base = String(key.dropLast(".lora_A".count))
      return (base, .a)
    }
    if key.hasSuffix(".lora_B") {
      let base = String(key.dropLast(".lora_B".count))
      return (base, .b)
    }

    // Alternative convention: .lora_A.weight / .lora_B.weight
    if key.contains(".lora_A.") {
      let base = key.replacingOccurrences(of: ".lora_A.", with: ".")
      return (base, .a)
    }
    if key.contains(".lora_B.") {
      let base = key.replacingOccurrences(of: ".lora_B.", with: ".")
      return (base, .b)
    }

    // Diffusers convention: .lora_down / .lora_up
    if key.hasSuffix(".lora_down") || key.contains(".lora_down.") {
      let base = key.replacingOccurrences(of: ".lora_down", with: "")
      return (base, .a)
    }
    if key.hasSuffix(".lora_up") || key.contains(".lora_up.") {
      let base = key.replacingOccurrences(of: ".lora_up", with: "")
      return (base, .b)
    }

    return (key, nil)
  }
}
