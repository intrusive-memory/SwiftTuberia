import Foundation
@preconcurrency import MLX
import MLXNN
import Tuberia

/// SDXL VAE Decoder: decodes 4-channel latents into pixel data.
///
/// Conforms to `Decoder` + `WeightedSegment`.
///
/// Expected input shape: [B, H/8, W/8, 4] (latent space)
/// Output: `DecodedOutput` with `ImageDecoderMetadata` and pixel data [B, H, W, 3]
///
/// Key mapping translates SDXL VAE safetensors keys (diffusers format) to
/// MLX module property paths. Tensor transposition converts NCHW → NHWC for
/// all 4D convolution weight tensors.
public final class SDXLVAEDecoder: Decoder, @unchecked Sendable {
  public typealias Configuration = SDXLVAEDecoderConfiguration

  private let configuration: Configuration
  private var model: SDXLVAEDecoderModel?
  private var _currentWeights: Tuberia.ModuleParameters?
  public private(set) var isLoaded: Bool = false

  public required init(configuration: Configuration) throws {
    self.configuration = configuration
  }

  // MARK: - Decoder Protocol

  public var expectedInputChannels: Int {
    configuration.latentChannels
  }

  public var scalingFactor: Float {
    configuration.scalingFactor
  }

  public func decode(_ latents: MLXArray) throws -> DecodedOutput {
    let shape = latents.shape

    // Validate input shape: [B, H, W, C] where C = latentChannels
    guard shape.count == 4 else {
      throw PipelineError.decodingFailed(
        reason: "SDXLVAEDecoder expects 4D input [B, H, W, C], got \(shape.count)D"
      )
    }

    guard shape[3] == configuration.latentChannels else {
      throw PipelineError.decodingFailed(
        reason: "SDXLVAEDecoder expects \(configuration.latentChannels) channels, got \(shape[3])"
      )
    }

    // Apply internal scaling: latents * (1.0 / scalingFactor)
    let scaledLatents = latents * (1.0 / configuration.scalingFactor)

    // Force evaluation before the VAE forward pass. The denoising loop calls eval(latents)
    // after each step, but that evaluation may leave the scaledLatents lazy. Evaluating
    // here ensures a clean, concrete input to the decoder — important because GroupNorm
    // reshapes inside the VAE mid-block can produce shapeless (ndim=0) tensors under
    // memory pressure if the input carries a stale lazy graph.
    eval(scaledLatents)

    // Run the real forward pass when model weights are loaded;
    // fall back to a correctly-shaped placeholder when the model is nil.
    let batchSize = shape[0]
    let latentH = shape[1]
    let latentW = shape[2]
    let outputH = latentH * 8
    let outputW = latentW * 8
    let pixelData: MLXArray
    if let loadedModel = model {
      do {
        pixelData = try withError { loadedModel(scaledLatents) }
      } catch {
        throw PipelineError.decodingFailed(
          reason: "SDXL VAE forward pass failed: \(error.localizedDescription)"
        )
      }
    } else {
      pixelData = placeholderForwardPass(
        scaledLatents, outputShape: [batchSize, outputH, outputW, 3])
    }

    let metadata = ImageDecoderMetadata(scalingFactor: configuration.scalingFactor)
    return DecodedOutput(data: pixelData, metadata: metadata)
  }

  // MARK: - WeightedSegment

  public var currentWeights: Tuberia.ModuleParameters? { _currentWeights }

  public var estimatedMemoryBytes: Int {
    // ~160 MB for fp16 SDXL VAE
    167_772_160
  }

  /// Maps SDXL VAE safetensors keys (diffusers format) to MLX module property paths.
  ///
  /// Safetensors format (diffusers):
  /// - `post_quant_conv.{weight,bias}` → pass through to `postQuantConv.{weight,bias}`
  /// - `decoder.conv_in.{weight,bias}` → `convIn.{weight,bias}`
  /// - `decoder.mid_block.resnets.{i}.{component}` → `midBlock.resnets.{i}.{component}`
  /// - `decoder.mid_block.attentions.0.{component}` → `midBlock.attention.{component}`
  /// - `decoder.up_blocks.{i}.resnets.{j}.{component}` → `upBlocks.{i}.resnets.{j}.{component}`
  /// - `decoder.up_blocks.{i}.upsamplers.0.conv.{weight,bias}` → `upBlocks.{i}.upsample.conv.{weight,bias}`
  /// - `decoder.conv_norm_out.{weight,bias}` → `convNormOut.{weight,bias}`
  /// - `decoder.conv_out.{weight,bias}` → `convOut.{weight,bias}`
  ///
  /// Filtered (return nil):
  /// - `encoder.*` — encoder weights not needed for decoding
  /// - `quant_conv.*` — encoder quantization conv not needed
  public var keyMapping: KeyMapping {
    { key in
      // Skip encoder-only keys
      if key.hasPrefix("encoder.") || key.hasPrefix("quant_conv.") {
        return nil
      }

      // post_quant_conv.{weight,bias} → postQuantConv.{weight,bias}
      if key.hasPrefix("post_quant_conv.") {
        let suffix = String(key.dropFirst("post_quant_conv.".count))
        return "postQuantConv.\(suffix)"
      }

      // All remaining keys should start with "decoder."
      guard key.hasPrefix("decoder.") else {
        return nil
      }

      let decoderKey = String(key.dropFirst("decoder.".count))

      // decoder.conv_in.{weight,bias} → convIn.{weight,bias}
      if decoderKey.hasPrefix("conv_in.") {
        let suffix = String(decoderKey.dropFirst("conv_in.".count))
        return "convIn.\(suffix)"
      }

      // decoder.conv_norm_out.{weight,bias} → convNormOut.{weight,bias}
      if decoderKey.hasPrefix("conv_norm_out.") {
        let suffix = String(decoderKey.dropFirst("conv_norm_out.".count))
        return "convNormOut.\(suffix)"
      }

      // decoder.conv_out.{weight,bias} → convOut.{weight,bias}
      if decoderKey.hasPrefix("conv_out.") {
        let suffix = String(decoderKey.dropFirst("conv_out.".count))
        return "convOut.\(suffix)"
      }

      // decoder.mid_block.* → midBlock.*
      if decoderKey.hasPrefix("mid_block.") {
        let midKey = String(decoderKey.dropFirst("mid_block.".count))
        return SDXLVAEDecoder.mapMidBlockKey(midKey)
      }

      // decoder.up_blocks.{i}.* → upBlocks.{i}.*
      if decoderKey.hasPrefix("up_blocks.") {
        let upKey = String(decoderKey.dropFirst("up_blocks.".count))
        return SDXLVAEDecoder.mapUpBlockKey(upKey)
      }

      return nil
    }
  }

  /// Transposes 4D convolution weight tensors from NCHW [out, in, kH, kW]
  /// to NHWC [out, kH, kW, in] as required by MLX Conv2d.
  ///
  /// Matches keys containing "conv" (case-insensitive) to handle both snake_case
  /// (`post_quant_conv`) and camelCase (`postQuantConv`) variants after key mapping.
  ///
  /// GroupNorm weight/bias (1D) and Linear weight/bias (2D) are passed through unchanged.
  public var tensorTransform: TensorTransform? {
    { key, tensor in
      // Transpose conv weights: [out, in, kH, kW] → [out, kH, kW, in]
      // Case-insensitive match to handle both "conv" and "Conv" (camelCase property names)
      if key.lowercased().contains("conv") && tensor.ndim == 4 {
        return tensor.transposed(0, 2, 3, 1)
      }
      return tensor
    }
  }

  /// Loads `ModuleParameters` into the `SDXLVAEDecoderModel` module tree.
  ///
  /// Instantiates `SDXLVAEDecoderModel` on first call, then uses
  /// `MLXNN.Module.update(parameters:)` to apply the flat parameter dictionary.
  /// The flat key format uses dot-separated Swift property names matching the
  /// module hierarchy (e.g. `midBlock.resnets.0.norm1.weight`).
  public func apply(weights: Tuberia.ModuleParameters) throws {
    // Instantiate the model if not already created
    if model == nil {
      model = SDXLVAEDecoderModel()
    }

    guard let loadedModel = model else {
      throw PipelineError.weightLoadingFailed(
        component: configuration.componentId,
        reason: "Failed to instantiate SDXLVAEDecoderModel"
      )
    }

    // Convert flat [String: MLXArray] to MLXNN.ModuleParameters (NestedDictionary)
    // using the dot-separated key paths that match the module hierarchy.
    let mlxParams = MLXNN.ModuleParameters.unflattened(weights.parameters)
    loadedModel.update(parameters: mlxParams)

    self._currentWeights = weights
    self.isLoaded = true
  }

  public func unload() {
    self.model = nil
    self._currentWeights = nil
    self.isLoaded = false
  }

  // MARK: - Private Key Mapping Helpers

  /// Maps mid_block subkeys to MLX property paths.
  ///
  /// - `resnets.{i}.{component}` → `resnets.{i}.{mlxComponent}`
  /// - `attentions.0.{component}` → `attention.{mlxComponent}`
  private static func mapMidBlockKey(_ key: String) -> String? {
    // mid_block.resnets.{i}.* → midBlock.resnets.{i}.*
    if key.hasPrefix("resnets.") {
      let resnetKey = String(key.dropFirst("resnets.".count))
      // resnetKey = "{i}.{component}"
      guard let dotIdx = resnetKey.firstIndex(of: ".") else { return nil }
      let indexStr = String(resnetKey[resnetKey.startIndex..<dotIdx])
      let component = String(resnetKey[resnetKey.index(after: dotIdx)...])
      let mappedComponent = mapResnetComponent(component)
      return "midBlock.resnets.\(indexStr).\(mappedComponent)"
    }

    // mid_block.attentions.0.* → midBlock.attention.*
    if key.hasPrefix("attentions.0.") {
      let attnComponent = String(key.dropFirst("attentions.0.".count))
      guard let mappedAttn = mapAttentionComponent(attnComponent) else { return nil }
      return "midBlock.attention.\(mappedAttn)"
    }

    return nil
  }

  /// Maps up_blocks subkeys to MLX property paths.
  ///
  /// - `{i}.resnets.{j}.{component}` → `upBlocks.{i}.resnets.{j}.{mlxComponent}`
  /// - `{i}.upsamplers.0.conv.{weight,bias}` → `upBlocks.{i}.upsample.conv.{weight,bias}`
  private static func mapUpBlockKey(_ key: String) -> String? {
    // key = "{i}.resnets.{j}.{component}" or "{i}.upsamplers.0.conv.{weight/bias}"
    guard let dotIdx = key.firstIndex(of: ".") else { return nil }
    let blockIndex = String(key[key.startIndex..<dotIdx])
    let remainder = String(key[key.index(after: dotIdx)...])

    // up_blocks.{i}.resnets.{j}.* → upBlocks.{i}.resnets.{j}.*
    if remainder.hasPrefix("resnets.") {
      let resnetKey = String(remainder.dropFirst("resnets.".count))
      guard let dot2 = resnetKey.firstIndex(of: ".") else { return nil }
      let resnetIndex = String(resnetKey[resnetKey.startIndex..<dot2])
      let component = String(resnetKey[resnetKey.index(after: dot2)...])
      let mappedComponent = mapResnetComponent(component)
      return "upBlocks.\(blockIndex).resnets.\(resnetIndex).\(mappedComponent)"
    }

    // up_blocks.{i}.upsamplers.0.conv.{weight,bias} → upBlocks.{i}.upsample.conv.{weight,bias}
    if remainder.hasPrefix("upsamplers.0.conv.") {
      let wbSuffix = String(remainder.dropFirst("upsamplers.0.conv.".count))
      return "upBlocks.\(blockIndex).upsample.conv.\(wbSuffix)"
    }

    return nil
  }

  /// Maps ResNet block component keys from diffusers format to MLX property names.
  ///
  /// Most components are identical (norm1, conv1, norm2, conv2).
  /// `conv_shortcut` becomes `convShortcut` (camelCase).
  private static func mapResnetComponent(_ component: String) -> String {
    switch component {
    case _ where component.hasPrefix("conv_shortcut."):
      let suffix = String(component.dropFirst("conv_shortcut.".count))
      return "convShortcut.\(suffix)"
    default:
      return component
    }
  }

  /// Maps attention block component keys from diffusers format to MLX property names.
  ///
  /// Diffusers uses `to_q`, `to_k`, `to_v`, `to_out.0`, `group_norm`.
  /// MLX module uses `query`, `key`, `value`, `projAttn`, `groupNorm`.
  private static func mapAttentionComponent(_ component: String) -> String? {
    if component.hasPrefix("group_norm.") {
      let suffix = String(component.dropFirst("group_norm.".count))
      return "groupNorm.\(suffix)"
    }
    if component.hasPrefix("to_q.") {
      let suffix = String(component.dropFirst("to_q.".count))
      return "query.\(suffix)"
    }
    if component.hasPrefix("to_k.") {
      let suffix = String(component.dropFirst("to_k.".count))
      return "key.\(suffix)"
    }
    if component.hasPrefix("to_v.") {
      let suffix = String(component.dropFirst("to_v.".count))
      return "value.\(suffix)"
    }
    if component.hasPrefix("to_out.0.") {
      let suffix = String(component.dropFirst("to_out.0.".count))
      return "projAttn.\(suffix)"
    }
    return nil
  }

  // MARK: - Placeholder

  /// Placeholder forward pass that produces correctly shaped output.
  /// Used when the model is not loaded.
  private func placeholderForwardPass(_ input: MLXArray, outputShape: [Int]) -> MLXArray {
    MLXArray.zeros(outputShape) + 0.5
  }
}
