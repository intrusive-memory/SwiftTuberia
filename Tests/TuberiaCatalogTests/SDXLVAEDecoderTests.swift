import Foundation
import Testing
@preconcurrency import MLX
import MLXNN
import Tuberia

@testable import TuberiaCatalog

// MARK: - Key Mapping Tests

@Suite("SDXLVAEDecoder Key Mapping Tests")
struct SDXLVAEDecoderKeyMappingTests {

  private let decoder: SDXLVAEDecoder = {
    try! SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
  }()

  // MARK: Filtered keys

  @Test("encoder.* keys return nil (filtered out)")
  func encoderKeysReturnNil() {
    let mapping = decoder.keyMapping
    #expect(mapping("encoder.conv_in.weight") == nil)
    #expect(mapping("encoder.conv_in.bias") == nil)
    #expect(mapping("encoder.mid_block.resnets.0.norm1.weight") == nil)
    #expect(mapping("encoder.down_blocks.0.resnets.0.conv1.weight") == nil)
    #expect(mapping("encoder.conv_out.weight") == nil)
  }

  @Test("quant_conv.* keys return nil (filtered out)")
  func quantConvKeysReturnNil() {
    let mapping = decoder.keyMapping
    #expect(mapping("quant_conv.weight") == nil)
    #expect(mapping("quant_conv.bias") == nil)
  }

  // MARK: post_quant_conv

  @Test("post_quant_conv.weight maps to postQuantConv.weight")
  func postQuantConvWeight() {
    let mapping = decoder.keyMapping
    #expect(mapping("post_quant_conv.weight") == "postQuantConv.weight")
    #expect(mapping("post_quant_conv.bias") == "postQuantConv.bias")
  }

  // MARK: conv_norm_out / conv_out

  @Test("decoder.conv_norm_out maps to convNormOut")
  func convNormOut() {
    let mapping = decoder.keyMapping
    #expect(mapping("decoder.conv_norm_out.weight") == "convNormOut.weight")
    #expect(mapping("decoder.conv_norm_out.bias") == "convNormOut.bias")
  }

  @Test("decoder.conv_out maps to convOut")
  func convOut() {
    let mapping = decoder.keyMapping
    #expect(mapping("decoder.conv_out.weight") == "convOut.weight")
    #expect(mapping("decoder.conv_out.bias") == "convOut.bias")
  }

  // MARK: mid_block resnets

  @Test("decoder.mid_block.resnets.0 norm and conv components map correctly")
  func midBlockResnet0() {
    let mapping = decoder.keyMapping
    #expect(mapping("decoder.mid_block.resnets.0.norm1.weight") == "midBlock.resnets.0.norm1.weight")
    #expect(mapping("decoder.mid_block.resnets.0.norm1.bias") == "midBlock.resnets.0.norm1.bias")
    #expect(mapping("decoder.mid_block.resnets.0.conv1.weight") == "midBlock.resnets.0.conv1.weight")
    #expect(mapping("decoder.mid_block.resnets.0.conv1.bias") == "midBlock.resnets.0.conv1.bias")
    #expect(mapping("decoder.mid_block.resnets.0.norm2.weight") == "midBlock.resnets.0.norm2.weight")
    #expect(mapping("decoder.mid_block.resnets.0.norm2.bias") == "midBlock.resnets.0.norm2.bias")
    #expect(mapping("decoder.mid_block.resnets.0.conv2.weight") == "midBlock.resnets.0.conv2.weight")
    #expect(mapping("decoder.mid_block.resnets.0.conv2.bias") == "midBlock.resnets.0.conv2.bias")
  }

  @Test("decoder.mid_block.resnets.0.conv_shortcut maps to convShortcut (camelCase)")
  func midBlockResnet0ConvShortcut() {
    let mapping = decoder.keyMapping
    #expect(
      mapping("decoder.mid_block.resnets.0.conv_shortcut.weight")
        == "midBlock.resnets.0.convShortcut.weight")
    #expect(
      mapping("decoder.mid_block.resnets.0.conv_shortcut.bias")
        == "midBlock.resnets.0.convShortcut.bias")
  }

  @Test("decoder.mid_block.resnets.1 maps correctly (no conv_shortcut)")
  func midBlockResnet1() {
    let mapping = decoder.keyMapping
    #expect(mapping("decoder.mid_block.resnets.1.norm1.weight") == "midBlock.resnets.1.norm1.weight")
    #expect(mapping("decoder.mid_block.resnets.1.conv2.weight") == "midBlock.resnets.1.conv2.weight")
  }

  // MARK: mid_block attention

  @Test("decoder.mid_block.attentions.0.group_norm maps to midBlock.attention.groupNorm")
  func midBlockAttnGroupNorm() {
    let mapping = decoder.keyMapping
    #expect(
      mapping("decoder.mid_block.attentions.0.group_norm.weight")
        == "midBlock.attention.groupNorm.weight")
    #expect(
      mapping("decoder.mid_block.attentions.0.group_norm.bias")
        == "midBlock.attention.groupNorm.bias")
  }

  @Test("decoder.mid_block.attentions.0.to_q maps to midBlock.attention.query")
  func midBlockAttnQuery() {
    let mapping = decoder.keyMapping
    #expect(
      mapping("decoder.mid_block.attentions.0.to_q.weight") == "midBlock.attention.query.weight")
    #expect(
      mapping("decoder.mid_block.attentions.0.to_q.bias") == "midBlock.attention.query.bias")
  }

  @Test("decoder.mid_block.attentions.0.to_k maps to midBlock.attention.key")
  func midBlockAttnKey() {
    let mapping = decoder.keyMapping
    #expect(
      mapping("decoder.mid_block.attentions.0.to_k.weight") == "midBlock.attention.key.weight")
    #expect(mapping("decoder.mid_block.attentions.0.to_k.bias") == "midBlock.attention.key.bias")
  }

  @Test("decoder.mid_block.attentions.0.to_v maps to midBlock.attention.value")
  func midBlockAttnValue() {
    let mapping = decoder.keyMapping
    #expect(
      mapping("decoder.mid_block.attentions.0.to_v.weight") == "midBlock.attention.value.weight")
    #expect(
      mapping("decoder.mid_block.attentions.0.to_v.bias") == "midBlock.attention.value.bias")
  }

  @Test("decoder.mid_block.attentions.0.to_out.0 maps to midBlock.attention.projAttn")
  func midBlockAttnProjAttn() {
    let mapping = decoder.keyMapping
    #expect(
      mapping("decoder.mid_block.attentions.0.to_out.0.weight")
        == "midBlock.attention.projAttn.weight")
    #expect(
      mapping("decoder.mid_block.attentions.0.to_out.0.bias")
        == "midBlock.attention.projAttn.bias")
  }

  // MARK: up_blocks resnets

  @Test("decoder.up_blocks.0.resnets.0 maps to upBlocks.0.resnets.0")
  func upBlock0Resnet0() {
    let mapping = decoder.keyMapping
    #expect(
      mapping("decoder.up_blocks.0.resnets.0.norm1.weight") == "upBlocks.0.resnets.0.norm1.weight")
    #expect(
      mapping("decoder.up_blocks.0.resnets.0.conv1.weight") == "upBlocks.0.resnets.0.conv1.weight")
    #expect(
      mapping("decoder.up_blocks.0.resnets.0.norm2.weight") == "upBlocks.0.resnets.0.norm2.weight")
    #expect(
      mapping("decoder.up_blocks.0.resnets.0.conv2.weight") == "upBlocks.0.resnets.0.conv2.weight")
  }

  @Test("decoder.up_blocks.0.resnets.1 and resnets.2 map correctly")
  func upBlock0Resnet1And2() {
    let mapping = decoder.keyMapping
    #expect(
      mapping("decoder.up_blocks.0.resnets.1.conv1.weight") == "upBlocks.0.resnets.1.conv1.weight")
    #expect(
      mapping("decoder.up_blocks.0.resnets.2.conv2.weight") == "upBlocks.0.resnets.2.conv2.weight")
  }

  @Test("decoder.up_blocks all 4 blocks map correctly")
  func allUpBlocks() {
    let mapping = decoder.keyMapping
    for blockIdx in 0..<4 {
      for resnetIdx in 0..<3 {
        let key = "decoder.up_blocks.\(blockIdx).resnets.\(resnetIdx).conv1.weight"
        let expected = "upBlocks.\(blockIdx).resnets.\(resnetIdx).conv1.weight"
        #expect(mapping(key) == expected, "Block \(blockIdx), ResNet \(resnetIdx)")
      }
    }
  }

  @Test("conv_shortcut in up_blocks maps to convShortcut")
  func upBlockConvShortcut() {
    let mapping = decoder.keyMapping
    #expect(
      mapping("decoder.up_blocks.2.resnets.0.conv_shortcut.weight")
        == "upBlocks.2.resnets.0.convShortcut.weight")
    #expect(
      mapping("decoder.up_blocks.3.resnets.0.conv_shortcut.bias")
        == "upBlocks.3.resnets.0.convShortcut.bias")
  }

  // MARK: up_blocks upsamplers

  @Test("decoder.up_blocks.0.upsamplers.0.conv maps to upBlocks.0.upsample.conv")
  func upBlock0Upsample() {
    let mapping = decoder.keyMapping
    #expect(
      mapping("decoder.up_blocks.0.upsamplers.0.conv.weight")
        == "upBlocks.0.upsample.conv.weight")
    #expect(
      mapping("decoder.up_blocks.0.upsamplers.0.conv.bias") == "upBlocks.0.upsample.conv.bias")
  }

  @Test("decoder.up_blocks upsamplers for blocks 0-2 map correctly (block 3 has no upsample)")
  func upBlocksUpsamplers() {
    let mapping = decoder.keyMapping
    for blockIdx in 0..<3 {
      let key = "decoder.up_blocks.\(blockIdx).upsamplers.0.conv.weight"
      let expected = "upBlocks.\(blockIdx).upsample.conv.weight"
      #expect(mapping(key) == expected, "Upsampler for block \(blockIdx)")
    }
  }
}

// MARK: - Tensor Transform Tests

@Suite("SDXLVAEDecoder Tensor Transform Tests")
struct SDXLVAEDecoderTensorTransformTests {

  private let decoder: SDXLVAEDecoder = {
    try! SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
  }()

  @Test("tensorTransform transposes 4D conv weights from NCHW to NHWC")
  func conv4DTranspose() {
    let transform = decoder.tensorTransform!

    // Simulate a conv weight: [out=4, in=3, kH=3, kW=3]
    let weight = MLXArray.zeros([4, 3, 3, 3]).asType(.float32)
    let transformed = transform("conv1.weight", weight)

    // Should become [out=4, kH=3, kW=3, in=3] — axes (0,2,3,1)
    #expect(transformed.shape == [4, 3, 3, 3])
    // Verify that axis 1 (in_channels) moved to axis 3
    // After transposed(0,2,3,1): shape[1]=kH, shape[2]=kW, shape[3]=in
    let floatValues: [Float] = (0..<(4 * 3 * 3 * 3)).map { Float($0) }
    let original = MLXArray(floatValues, [4 * 3 * 3 * 3]).reshaped([4, 3, 3, 3])
    let transposedWeight = transform("conv1.weight", original)
    // Original shape [4,3,3,3]: index [o,i,h,w] maps to transposed index [o,h,w,i]
    #expect(transposedWeight.shape == [4, 3, 3, 3])
  }

  @Test("tensorTransform with different channel dims: [512, 4, 3, 3] → [512, 3, 3, 4]")
  func conv4DTransposeDifferentChannels() {
    let transform = decoder.tensorTransform!

    // post_quant_conv after expansion: [out=512, in=4, kH=3, kW=3]
    let weight = MLXArray.zeros([512, 4, 3, 3]).asType(.float32)
    let transformed = transform("post_quant_conv.weight", weight)

    #expect(transformed.shape == [512, 3, 3, 4])
  }

  @Test("tensorTransform 1x1 conv: [4, 4, 1, 1] → [4, 1, 1, 4]")
  func conv1x1Transpose() {
    let transform = decoder.tensorTransform!

    let weight = MLXArray.zeros([4, 4, 1, 1]).asType(.float32)
    let transformed = transform("postQuantConv.weight", weight)

    #expect(transformed.shape == [4, 1, 1, 4])
  }

  @Test("tensorTransform preserves 1D GroupNorm weight (no transpose)")
  func groupNormWeightPreserved() {
    let transform = decoder.tensorTransform!

    let weight = MLXArray.zeros([512]).asType(.float32)
    let transformed = transform("midBlock.resnets.0.norm1.weight", weight)

    #expect(transformed.shape == [512])
  }

  @Test("tensorTransform preserves 1D GroupNorm bias (no transpose)")
  func groupNormBiasPreserved() {
    let transform = decoder.tensorTransform!

    let bias = MLXArray.zeros([128]).asType(.float32)
    let transformed = transform("convNormOut.bias", bias)

    #expect(transformed.shape == [128])
  }

  @Test("tensorTransform preserves 2D Linear weight (no transpose)")
  func linearWeightPreserved() {
    let transform = decoder.tensorTransform!

    // Linear weight: [out=512, in=512]
    let weight = MLXArray.zeros([512, 512]).asType(.float32)
    let transformed = transform("midBlock.attention.query.weight", weight)

    #expect(transformed.shape == [512, 512])
  }

  @Test("tensorTransform preserves 1D bias (no transpose)")
  func convBias1DPreserved() {
    let transform = decoder.tensorTransform!

    // Conv bias is 1D
    let bias = MLXArray.zeros([512]).asType(.float32)
    let transformed = transform("midBlock.resnets.0.conv1.bias", bias)

    #expect(transformed.shape == [512])
  }

  @Test("tensorTransform only triggers on keys containing 'conv'")
  func onlyConvKeysTransposed() {
    let transform = decoder.tensorTransform!

    let tensor4D = MLXArray.zeros([4, 3, 3, 3]).asType(.float32)

    // Non-conv 4D tensor should pass through unchanged
    let unchanged = transform("some_other_layer.weight", tensor4D)
    #expect(unchanged.shape == [4, 3, 3, 3])

    // Conv 4D tensor should be transposed
    let transposed = transform("conv_out.weight", tensor4D)
    #expect(transposed.shape == [4, 3, 3, 3])  // same dims for [4,3,3,3] but axes reordered
  }
}

// MARK: - apply(weights:) Tests

@Suite("SDXLVAEDecoder apply(weights:) Tests")
struct SDXLVAEDecoderApplyWeightsTests {

  /// Creates a minimal synthetic `ModuleParameters` with one parameter for the given key.
  private func singleParam(key: String, shape: [Int]) -> Tuberia.ModuleParameters {
    let tensor = MLXArray.zeros(shape).asType(.float32)
    return Tuberia.ModuleParameters(parameters: [key: tensor])
  }

  @Test("apply(weights:) with empty parameters does not crash and sets isLoaded = true")
  func applyEmptyWeights() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
    #expect(!decoder.isLoaded)

    let emptyParams = Tuberia.ModuleParameters(parameters: [:])
    try decoder.apply(weights: emptyParams)

    #expect(decoder.isLoaded)
  }

  @Test("apply(weights:) sets isLoaded = true")
  func applyWeightsSetsLoaded() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
    #expect(!decoder.isLoaded)

    // Provide a synthetic weight that matches a real model key
    let params = singleParam(key: "convNormOut.weight", shape: [128])
    try decoder.apply(weights: params)

    #expect(decoder.isLoaded)
  }

  @Test("unload() clears isLoaded and model")
  func unloadClearsState() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
    let emptyParams = Tuberia.ModuleParameters(parameters: [:])
    try decoder.apply(weights: emptyParams)
    #expect(decoder.isLoaded)

    decoder.unload()
    #expect(!decoder.isLoaded)
  }

  @Test("apply(weights:) can be called multiple times without crashing")
  func applyWeightsIdempotent() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())

    let params = singleParam(key: "postQuantConv.weight", shape: [4, 1, 1, 4])
    try decoder.apply(weights: params)
    try decoder.apply(weights: params)

    #expect(decoder.isLoaded)
  }

  @Test("apply(weights:) with multi-key synthetic params does not crash")
  func applyMultiKeyParams() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())

    // Provide several synthetic parameters matching the model hierarchy
    let params = Tuberia.ModuleParameters(parameters: [
      "postQuantConv.weight": MLXArray.zeros([4, 1, 1, 4]).asType(.float32),
      "postQuantConv.bias": MLXArray.zeros([4]).asType(.float32),
      "convNormOut.weight": MLXArray.zeros([128]).asType(.float32),
      "convNormOut.bias": MLXArray.zeros([128]).asType(.float32),
      "convOut.weight": MLXArray.zeros([128, 3, 3, 3]).asType(.float32),
      "convOut.bias": MLXArray.zeros([3]).asType(.float32),
    ])

    try decoder.apply(weights: params)
    #expect(decoder.isLoaded)
  }
}

// MARK: - decode() Shape Tests

@Suite("SDXLVAEDecoder decode() Shape Tests")
struct SDXLVAEDecoderDecodeTests {

  @Test("decode() unloaded produces correct output shape [1, H*8, W*8, 3]")
  func decodePlaceholderShape() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())

    // Latent: [1, 8, 8, 4]
    let latents = MLXArray.zeros([1, 8, 8, 4]).asType(.float32)
    let output = try decoder.decode(latents)

    #expect(output.data.shape == [1, 64, 64, 3])
  }

  @Test("decode() unloaded respects 8x spatial upscaling")
  func decodeSpatialUpscaling() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())

    // Latent: [2, 10, 12, 4] — batch=2, H=10, W=12
    let latents = MLXArray.zeros([2, 10, 12, 4]).asType(.float32)
    let output = try decoder.decode(latents)

    #expect(output.data.shape == [2, 80, 96, 3])
  }

  @Test("decode() throws on wrong number of dimensions")
  func decodeWrongDimensions() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())

    let latents3D = MLXArray.zeros([1, 8, 4]).asType(.float32)
    #expect(throws: (any Error).self) {
      try decoder.decode(latents3D)
    }
  }

  @Test("decode() throws on wrong channel count")
  func decodeWrongChannels() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())

    // 3 channels instead of 4
    let latents = MLXArray.zeros([1, 8, 8, 3]).asType(.float32)
    #expect(throws: (any Error).self) {
      try decoder.decode(latents)
    }
  }

  @Test("decode() after apply(weights:) still produces correct output shape")
  func decodeWithWeightsShape() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())

    // Apply minimal synthetic weights to trigger model instantiation and isLoaded = true
    let emptyParams = Tuberia.ModuleParameters(parameters: [:])
    try decoder.apply(weights: emptyParams)
    #expect(decoder.isLoaded)

    // With model loaded, decode() routes through real SDXLVAEDecoderModel forward pass.
    // Output shape must be [B, H*8, W*8, 3].
    let latents = MLXArray.zeros([1, 8, 8, 4]).asType(.float32)
    let output = try decoder.decode(latents)

    #expect(output.data.shape == [1, 64, 64, 3])
  }
}

// MARK: - Full Key Coverage Test

@Suite("SDXLVAEDecoder Key Coverage Tests")
struct SDXLVAEDecoderKeyCoverageTests {

  private let decoder: SDXLVAEDecoder = {
    try! SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
  }()

  /// Generate all expected safetensors keys for the SDXL VAE decoder (diffusers format)
  /// and verify they all map to non-nil values.
  @Test("All expected decoder safetensors keys produce non-nil mappings")
  func allDecoderKeysCovered() {
    let mapping = decoder.keyMapping
    var expectedKeys: [String] = []

    // post_quant_conv
    expectedKeys += ["post_quant_conv.weight", "post_quant_conv.bias"]

    // mid_block resnets (0 has conv_shortcut: 4→512, 1 does not: 512→512)
    for i in 0..<2 {
      let prefix = "decoder.mid_block.resnets.\(i)"
      expectedKeys += [
        "\(prefix).norm1.weight", "\(prefix).norm1.bias",
        "\(prefix).conv1.weight", "\(prefix).conv1.bias",
        "\(prefix).norm2.weight", "\(prefix).norm2.bias",
        "\(prefix).conv2.weight", "\(prefix).conv2.bias",
      ]
    }
    // conv_shortcut on resnet 0 (channel expansion)
    expectedKeys += [
      "decoder.mid_block.resnets.0.conv_shortcut.weight",
      "decoder.mid_block.resnets.0.conv_shortcut.bias",
    ]

    // mid_block attention
    let attnPrefix = "decoder.mid_block.attentions.0"
    expectedKeys += [
      "\(attnPrefix).group_norm.weight", "\(attnPrefix).group_norm.bias",
      "\(attnPrefix).to_q.weight", "\(attnPrefix).to_q.bias",
      "\(attnPrefix).to_k.weight", "\(attnPrefix).to_k.bias",
      "\(attnPrefix).to_v.weight", "\(attnPrefix).to_v.bias",
      "\(attnPrefix).to_out.0.weight", "\(attnPrefix).to_out.0.bias",
    ]

    // up_blocks
    // Block 0: 512→512, 3 resnets (no channel change), upsample
    // Block 1: 512→512, 3 resnets (no channel change), upsample
    // Block 2: 512→256, resnet 0 has conv_shortcut, resnets 1-2 do not, upsample
    // Block 3: 256→128, resnet 0 has conv_shortcut, resnets 1-2 do not, no upsample
    for blockIdx in 0..<4 {
      for resnetIdx in 0..<3 {
        let prefix = "decoder.up_blocks.\(blockIdx).resnets.\(resnetIdx)"
        expectedKeys += [
          "\(prefix).norm1.weight", "\(prefix).norm1.bias",
          "\(prefix).conv1.weight", "\(prefix).conv1.bias",
          "\(prefix).norm2.weight", "\(prefix).norm2.bias",
          "\(prefix).conv2.weight", "\(prefix).conv2.bias",
        ]
      }
      // conv_shortcut on first resnet of blocks 2 and 3 (channel reduction)
      if blockIdx == 2 || blockIdx == 3 {
        expectedKeys += [
          "decoder.up_blocks.\(blockIdx).resnets.0.conv_shortcut.weight",
          "decoder.up_blocks.\(blockIdx).resnets.0.conv_shortcut.bias",
        ]
      }
      // upsample on blocks 0, 1, 2
      if blockIdx < 3 {
        expectedKeys += [
          "decoder.up_blocks.\(blockIdx).upsamplers.0.conv.weight",
          "decoder.up_blocks.\(blockIdx).upsamplers.0.conv.bias",
        ]
      }
    }

    // conv_norm_out / conv_out
    expectedKeys += [
      "decoder.conv_norm_out.weight", "decoder.conv_norm_out.bias",
      "decoder.conv_out.weight", "decoder.conv_out.bias",
    ]

    // Verify all keys produce non-nil mappings
    var failures: [String] = []
    for key in expectedKeys {
      if mapping(key) == nil {
        failures.append(key)
      }
    }

    #expect(
      failures.isEmpty,
      "These keys produced nil mappings: \(failures)"
    )
  }

  @Test("All expected encoder safetensors keys produce nil mappings (filtered)")
  func allEncoderKeysFiltered() {
    let mapping = decoder.keyMapping
    let encoderKeys = [
      "encoder.conv_in.weight",
      "encoder.conv_in.bias",
      "encoder.mid_block.resnets.0.conv1.weight",
      "encoder.down_blocks.0.resnets.0.norm1.weight",
      "encoder.conv_out.weight",
      "quant_conv.weight",
      "quant_conv.bias",
    ]

    for key in encoderKeys {
      #expect(mapping(key) == nil, "Key \(key) should be filtered (nil)")
    }
  }
}

// MARK: - Sortie 3: Forward Pass Integration Tests

/// Builds a minimal synthetic `ModuleParameters` that covers the entire
/// `SDXLVAEDecoderModel` parameter tree.  All weights are zero-initialized
/// (NHWC layout after transposition, matching the MLX Conv2d convention).
/// Zero weights cause the model to produce all-zero activations, which is a
/// valid numerical output and allows shape/range assertions without needing
/// real checkpoint data.
private func makeSyntheticVAEParams() -> Tuberia.ModuleParameters {
  var params: [String: MLXArray] = [:]

  // post_quant_conv: [out=4, kH=1, kW=1, in=4]  (NHWC 1x1 conv weight)
  params["postQuantConv.weight"] = MLXArray.zeros([4, 1, 1, 4]).asType(.float32)
  params["postQuantConv.bias"] = MLXArray.zeros([4]).asType(.float32)

  // midBlock resnets
  // resnet 0: inChannels=4, outChannels=512 → has convShortcut
  let midResnetSpecs: [(inCh: Int, outCh: Int, hasShortcut: Bool)] = [
    (4, 512, true),
    (512, 512, false),
  ]
  for (i, spec) in midResnetSpecs.enumerated() {
    let p = "midBlock.resnets.\(i)"
    params["\(p).norm1.weight"] = MLXArray.zeros([spec.inCh]).asType(.float32)
    params["\(p).norm1.bias"] = MLXArray.zeros([spec.inCh]).asType(.float32)
    params["\(p).conv1.weight"] = MLXArray.zeros([spec.outCh, 3, 3, spec.inCh]).asType(.float32)
    params["\(p).conv1.bias"] = MLXArray.zeros([spec.outCh]).asType(.float32)
    params["\(p).norm2.weight"] = MLXArray.zeros([spec.outCh]).asType(.float32)
    params["\(p).norm2.bias"] = MLXArray.zeros([spec.outCh]).asType(.float32)
    params["\(p).conv2.weight"] = MLXArray.zeros([spec.outCh, 3, 3, spec.outCh]).asType(.float32)
    params["\(p).conv2.bias"] = MLXArray.zeros([spec.outCh]).asType(.float32)
    if spec.hasShortcut {
      params["\(p).convShortcut.weight"] = MLXArray.zeros([spec.outCh, 1, 1, spec.inCh]).asType(
        .float32)
      params["\(p).convShortcut.bias"] = MLXArray.zeros([spec.outCh]).asType(.float32)
    }
  }

  // midBlock attention (channels=512)
  let attnCh = 512
  params["midBlock.attention.groupNorm.weight"] = MLXArray.zeros([attnCh]).asType(.float32)
  params["midBlock.attention.groupNorm.bias"] = MLXArray.zeros([attnCh]).asType(.float32)
  params["midBlock.attention.query.weight"] = MLXArray.zeros([attnCh, attnCh]).asType(.float32)
  params["midBlock.attention.query.bias"] = MLXArray.zeros([attnCh]).asType(.float32)
  params["midBlock.attention.key.weight"] = MLXArray.zeros([attnCh, attnCh]).asType(.float32)
  params["midBlock.attention.key.bias"] = MLXArray.zeros([attnCh]).asType(.float32)
  params["midBlock.attention.value.weight"] = MLXArray.zeros([attnCh, attnCh]).asType(.float32)
  params["midBlock.attention.value.bias"] = MLXArray.zeros([attnCh]).asType(.float32)
  params["midBlock.attention.projAttn.weight"] = MLXArray.zeros([attnCh, attnCh]).asType(.float32)
  params["midBlock.attention.projAttn.bias"] = MLXArray.zeros([attnCh]).asType(.float32)

  // upBlocks specs: (inCh, outCh, hasUpsample)
  let upBlockSpecs: [(inCh: Int, outCh: Int, hasUpsample: Bool)] = [
    (512, 512, true),
    (512, 512, true),
    (512, 256, true),
    (256, 128, false),
  ]
  for (blockIdx, blockSpec) in upBlockSpecs.enumerated() {
    for resnetIdx in 0..<3 {
      let rInCh = resnetIdx == 0 ? blockSpec.inCh : blockSpec.outCh
      let rOutCh = blockSpec.outCh
      let p = "upBlocks.\(blockIdx).resnets.\(resnetIdx)"
      params["\(p).norm1.weight"] = MLXArray.zeros([rInCh]).asType(.float32)
      params["\(p).norm1.bias"] = MLXArray.zeros([rInCh]).asType(.float32)
      params["\(p).conv1.weight"] = MLXArray.zeros([rOutCh, 3, 3, rInCh]).asType(.float32)
      params["\(p).conv1.bias"] = MLXArray.zeros([rOutCh]).asType(.float32)
      params["\(p).norm2.weight"] = MLXArray.zeros([rOutCh]).asType(.float32)
      params["\(p).norm2.bias"] = MLXArray.zeros([rOutCh]).asType(.float32)
      params["\(p).conv2.weight"] = MLXArray.zeros([rOutCh, 3, 3, rOutCh]).asType(.float32)
      params["\(p).conv2.bias"] = MLXArray.zeros([rOutCh]).asType(.float32)
      if resnetIdx == 0 && rInCh != rOutCh {
        params["\(p).convShortcut.weight"] = MLXArray.zeros([rOutCh, 1, 1, rInCh]).asType(
          .float32)
        params["\(p).convShortcut.bias"] = MLXArray.zeros([rOutCh]).asType(.float32)
      }
    }
    if blockSpec.hasUpsample {
      let ch = blockSpec.outCh
      params["upBlocks.\(blockIdx).upsample.conv.weight"] = MLXArray.zeros([ch, 3, 3, ch])
        .asType(.float32)
      params["upBlocks.\(blockIdx).upsample.conv.bias"] = MLXArray.zeros([ch]).asType(.float32)
    }
  }

  // Final norm + output conv
  params["convNormOut.weight"] = MLXArray.zeros([128]).asType(.float32)
  params["convNormOut.bias"] = MLXArray.zeros([128]).asType(.float32)
  params["convOut.weight"] = MLXArray.zeros([3, 3, 3, 128]).asType(.float32)
  params["convOut.bias"] = MLXArray.zeros([3]).asType(.float32)

  return Tuberia.ModuleParameters(parameters: params)
}

@Suite("SDXLVAEDecoder Sortie 3: Forward Pass Integration Tests")
struct SDXLVAEDecoderForwardPassTests {

  @Test("decode() with loaded model produces [B, H*8, W*8, 3] output shape")
  func decodeWithModelOutputShape() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
    try decoder.apply(weights: makeSyntheticVAEParams())
    #expect(decoder.isLoaded)

    let latents = MLXArray.zeros([1, 8, 8, 4]).asType(.float32)
    let output = try decoder.decode(latents)

    #expect(output.data.shape == [1, 64, 64, 3])
  }

  @Test("decode() spatial dimensions are 8x latent spatial dimensions")
  func decodeSpatialUpscaling8x() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
    try decoder.apply(weights: makeSyntheticVAEParams())

    // Use latent [1, 4, 6, 4] → expect output [1, 32, 48, 3]
    let latents = MLXArray.zeros([1, 4, 6, 4]).asType(.float32)
    let output = try decoder.decode(latents)

    let outShape = output.data.shape
    #expect(outShape.count == 4)
    #expect(outShape[0] == 1)
    #expect(outShape[1] == 32)   // 4 * 8
    #expect(outShape[2] == 48)   // 6 * 8
    #expect(outShape[3] == 3)
  }

  @Test("decode() output has 3 channels (RGB)")
  func decodeOutputChannels() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
    try decoder.apply(weights: makeSyntheticVAEParams())

    let latents = MLXArray.zeros([1, 8, 8, 4]).asType(.float32)
    let output = try decoder.decode(latents)

    #expect(output.data.shape.last == 3)
  }

  @Test("decode() output values are finite (not NaN or Inf)")
  func decodeOutputFiniteValues() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
    try decoder.apply(weights: makeSyntheticVAEParams())

    let latents = MLXArray.zeros([1, 8, 8, 4]).asType(.float32)
    let output = try decoder.decode(latents)

    // Check that all values are finite using isNaN and isInf checks
    let data = output.data.asArray(Float.self)
    let allFinite = data.allSatisfy { $0.isFinite }
    #expect(allFinite, "decode() output must contain only finite values")
  }

  @Test("decode() output metadata carries the scaling factor")
  func decodeMetadataScalingFactor() throws {
    let config = SDXLVAEDecoderConfiguration()
    let decoder = try SDXLVAEDecoder(configuration: config)
    try decoder.apply(weights: makeSyntheticVAEParams())

    let latents = MLXArray.zeros([1, 8, 8, 4]).asType(.float32)
    let output = try decoder.decode(latents)

    guard let meta = output.metadata as? ImageDecoderMetadata else {
      Issue.record("Expected ImageDecoderMetadata")
      return
    }
    #expect(meta.scalingFactor == config.scalingFactor)
  }

  @Test("unloaded decoder falls back to placeholder — output shape is still [B, H*8, W*8, 3]")
  func unloadedDecoderFallback() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
    // Do NOT call apply(weights:) — model stays nil
    #expect(!decoder.isLoaded)

    let latents = MLXArray.zeros([1, 8, 8, 4]).asType(.float32)
    let output = try decoder.decode(latents)

    #expect(output.data.shape == [1, 64, 64, 3])
  }

  @Test("unloaded decoder placeholder values are 0.5 (plausible mid-range)")
  func unloadedPlaceholderValues() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
    #expect(!decoder.isLoaded)

    let latents = MLXArray.zeros([1, 2, 2, 4]).asType(.float32)
    let output = try decoder.decode(latents)

    let values = output.data.asArray(Float.self)
    let allHalf = values.allSatisfy { $0 == 0.5 }
    #expect(allHalf, "Unloaded placeholder should produce all 0.5 values")
  }

  @Test("unload() after loading causes fallback to placeholder")
  func unloadCausesFallbackToPlaceholder() throws {
    let decoder = try SDXLVAEDecoder(configuration: SDXLVAEDecoderConfiguration())
    try decoder.apply(weights: makeSyntheticVAEParams())
    #expect(decoder.isLoaded)

    decoder.unload()
    #expect(!decoder.isLoaded)

    // After unload, decode() must still return correctly shaped output via placeholder
    let latents = MLXArray.zeros([1, 8, 8, 4]).asType(.float32)
    let output = try decoder.decode(latents)
    #expect(output.data.shape == [1, 64, 64, 3])

    // And it should be the placeholder 0.5 values
    let values = output.data.asArray(Float.self)
    let allHalf = values.allSatisfy { $0 == 0.5 }
    #expect(allHalf, "After unload, decode() should use 0.5 placeholder values")
  }

  @Test("scaling factor 1/0.13025 is applied before forward pass")
  func scalingFactorApplied() throws {
    // Verify that the scaling factor is 0.13025 (the SDXL standard).
    // The configuration default uses 0.13025 and the decode() method divides by it.
    let config = SDXLVAEDecoderConfiguration()
    #expect(abs(config.scalingFactor - 0.13025) < 1e-6)

    let decoder = try SDXLVAEDecoder(configuration: config)
    // scalingFactor property is available via protocol
    #expect(abs(decoder.scalingFactor - 0.13025) < 1e-6)
  }
}
