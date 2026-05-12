# SwiftTuberia — Inference Implementation Requirements

**Parent**: [`REQUIREMENTS.md`](../REQUIREMENTS.md) — SwiftTuberia Overview
**Scope**: Replace all placeholder/stub implementations with real neural network forward passes, tokenizer integration, LoRA wiring, and img2img support. This document covers every gap between the current codebase and a working inference pipeline.
**Status**: DRAFT
**Date**: 2026-03-26
**Predecessor**: [Execution Plan 01](../docs/complete/swifttuberia-pipeline-system-01-execution-plan.md) (pipeline plumbing — complete)

---

## Context

Execution Plan 01 built the pipeline plumbing: protocols, orchestration, weight loading, memory management, schedulers, renderers, and catalog registration. All of this works correctly.

What remains are the **neural network forward passes** and **integration wiring** — the code that turns loaded weights into actual inference. Five areas need implementation:

| # | Area | Current State | Target State |
|---|------|---------------|--------------|
| 1 | T5-XXL Encoder architecture | Returns zero embeddings | Real T5 transformer producing 4096-dim embeddings |
| 2 | T5-XXL Tokenizer loading | Not loaded | `AutoTokenizer.from(modelFolder:)` via Acervo |
| 3 | SDXL VAE Decoder architecture | Returns 0.5 for all pixels | Real VAE decoder (ResNet + attention + upsample) |
| 4 | LoRA application in DiffusionPipeline | Loaded then discarded | Merged into backbone weights before generation, restored after |
| 5 | CGImage-to-MLXArray for img2img | Returns zeros | Real pixel conversion from CGImage input |

Secondary issues (lower priority):

| # | Area | Current State | Target State |
|---|------|---------------|--------------|
| 6 | T5-XXL key mapping | Identity passthrough | Real ~580-key mapping for T5 encoder architecture |
| 7 | SDXL VAE key mapping | Identity passthrough | Real ~130-key mapping for SDXL VAE architecture |
| 8 | FlowMatchEulerScheduler fallback | Returns unchanged sample on missing timestep | Graceful interpolation or clear error |

---

## INF-1: T5-XXL Transformer Encoder Architecture

**File**: `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift`
**New file**: `Sources/TuberiaCatalog/Encoders/T5TransformerEncoder.swift`
**Priority**: P0 (without this, no text conditioning works)

### Architecture Specification

T5-XXL encoder-only model with the following structure:

```
Token Embedding (vocab_size=32128, dim=4096)
  |
  v
24x Transformer Encoder Blocks:
  ├── RMS Layer Norm (pre-attention)
  ├── Self-Attention (64 heads, head_dim=64, with relative position bias)
  ├── Residual connection
  ├── RMS Layer Norm (pre-FFN)
  ├── Gated Feed-Forward (GeGLU activation, inner_dim=10240)
  └── Residual connection
  |
  v
Final RMS Layer Norm
  |
  v
Output embeddings [B, seq_len, 4096]
```

### Sub-components to implement

Each of the following should be an `MLXNN.Module` subclass in `T5TransformerEncoder.swift`:

1. **`T5RMSNorm`** — RMS layer normalization (no bias, no mean subtraction)
   - Parameters: `weight` [hidden_dim]
   - Forward: `x * rsqrt(mean(x^2) + eps) * weight`, eps=1e-6

2. **`T5Attention`** — Multi-head self-attention with relative position bias
   - Parameters: `q` [hidden_dim, hidden_dim], `k` [hidden_dim, hidden_dim], `v` [hidden_dim, hidden_dim], `o` [hidden_dim, hidden_dim]
   - No bias terms (T5 uses bias-free attention projections)
   - Relative position bias: shared per-layer bucket table, 32 buckets, bidirectional
   - `relative_attention_bias` parameter: [num_heads, num_buckets] — **only on layer 0**, shared across all layers via reference

3. **`T5GatedFFN`** — Gated feed-forward with GeGLU
   - Parameters: `wi_0` [hidden_dim, ffn_dim], `wi_1` [hidden_dim, ffn_dim], `wo` [ffn_dim, hidden_dim]
   - No bias terms
   - Forward: `wo(gelu(wi_0(x)) * wi_1(x))`

4. **`T5EncoderBlock`** — Single transformer block
   - Contains: T5RMSNorm (pre-attn), T5Attention, T5RMSNorm (pre-ffn), T5GatedFFN
   - Residual connections around attention and FFN

5. **`T5TransformerEncoder`** — Full encoder stack
   - Token embedding: `MLXNN.Embedding(32128, 4096)`
   - 24 encoder blocks
   - Final RMS norm
   - Forward: embed -> blocks -> final_norm -> output

### Weight key mapping (INF-6)

The T5 safetensors use hierarchical keys. The mapping must translate:

```
shared.weight                                    -> embedding.weight
encoder.block.{i}.layer.0.layer_norm.weight      -> blocks.{i}.pre_attn_norm.weight
encoder.block.{i}.layer.0.SelfAttention.q.weight -> blocks.{i}.attention.q.weight
encoder.block.{i}.layer.0.SelfAttention.k.weight -> blocks.{i}.attention.k.weight
encoder.block.{i}.layer.0.SelfAttention.v.weight -> blocks.{i}.attention.v.weight
encoder.block.{i}.layer.0.SelfAttention.o.weight -> blocks.{i}.attention.o.weight
encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight -> relative_position_bias.weight
encoder.block.{i}.layer.1.layer_norm.weight      -> blocks.{i}.pre_ffn_norm.weight
encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight -> blocks.{i}.ffn.wi_0.weight
encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight -> blocks.{i}.ffn.wi_1.weight
encoder.block.{i}.layer.1.DenseReluDense.wo.weight   -> blocks.{i}.ffn.wo.weight
encoder.final_layer_norm.weight                  -> final_norm.weight
```

This produces ~580 keys (24 layers x 11 keys/layer + embedding + final norm + relative position bias).

### Tokenizer loading

In `T5XXLEncoder.init(configuration:)`:

```swift
// Load tokenizer from the Acervo component directory
let tokenizer = try await AcervoManager.shared.withModelAccess(configuration.componentId) { dir in
    try AutoTokenizer.from(modelFolder: dir.path)
}
```

Store as `private var tokenizer: (any Tokenizer)?`. The `encode()` method uses it to tokenize input text before running the transformer.

### `apply(weights:)` implementation

Must load each sub-module's parameters from `ModuleParameters`:
- Embedding table
- 24 blocks (attention Q/K/V/O, two norms, FFN wi_0/wi_1/wo)
- Relative position bias (from layer 0)
- Final layer norm

Use `MLXNN.Module.update(parameters:)` or manual parameter assignment.

### Exit criteria

- [ ] `encode("a photo of a cat")` with loaded weights produces `[1, seqLen, 4096]` non-zero, non-constant embeddings
- [ ] Attention mask correctly marks real tokens as 1 and padding as 0
- [ ] Relative position bias is computed correctly (bucket-based, bidirectional)
- [ ] Unloaded encoder still produces correctly-shaped placeholder output (for testing)
- [ ] Memory estimate matches actual loaded weight size within 10%
- [ ] Unit test: synthetic weights -> deterministic output shape and non-trivial values
- [ ] Integration test (gated `#if INTEGRATION_TESTS`): real T5-XXL int4 weights -> embedding PSNR against reference

---

## INF-2: T5-XXL Tokenizer Integration

**File**: `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift`
**Priority**: P0 (required by INF-1)

### Requirements

1. `T5XXLEncoder.init(configuration:)` must become `async throws` (or tokenizer loading deferred to first `encode()` call) to support Acervo's async `withModelAccess`
2. Tokenizer loaded via `swift-transformers` `AutoTokenizer.from(modelFolder:)` pointing at the Acervo component directory
3. The tokenizer files (`tokenizer.json`, `tokenizer_config.json`) are bundled in the same Acervo component as the weights (component ID: `t5-xxl-encoder-int4`)
4. Tokenization: text -> token IDs -> clamped to `maxSequenceLength` -> padded with pad token (ID 0) if shorter
5. Attention mask derived from tokenization: 1 for real tokens, 0 for padding

### Design consideration: async init

The `TextEncoder` protocol defines `init(configuration:) throws`. Loading a tokenizer from Acervo requires async. Options:

**Option A**: Lazy tokenizer loading — defer to first `encode()` call. Store `tokenizer` as optional, load on first use. Simpler protocol compatibility but first encode has higher latency.

**Option B**: Separate `loadTokenizer()` async method called during `DiffusionPipeline.loadModels()` alongside weight loading. More explicit lifecycle control.

**Recommendation**: Option B — add a `loadTokenizer()` step in the pipeline's load phase. This keeps `init` synchronous and makes the async boundary explicit.

### Exit criteria

- [ ] Tokenizer successfully loads from Acervo component directory
- [ ] "a photo of a cat" tokenizes to a plausible token count (4-7 tokens)
- [ ] Padding produces correct attention mask
- [ ] `maxSequenceLength` is respected (truncation works)
- [ ] Empty string tokenizes to a single EOS/pad token

---

## INF-3: SDXL VAE Decoder Architecture

**File**: `Sources/TuberiaCatalog/Decoders/SDXLVAEDecoder.swift`
**New file**: `Sources/TuberiaCatalog/Decoders/SDXLVAEModel.swift`
**Priority**: P0 (without this, latents cannot be decoded to images)

### Architecture Specification

The SDXL VAE decoder transforms 4-channel latents into RGB pixel data:

```
Input: [B, H/8, W/8, 4] (latent space, after 1/scalingFactor applied)
  |
post_quant_conv: Conv2d(4, 4, kernel=1)
  |
Mid Block:
  ├── ResnetBlock2D(4 -> 512)       (Note: first ResNet block expands channels)
  ├── AttentionBlock(512, 1 head)
  └── ResnetBlock2D(512 -> 512)
  |
Up Block 0: (512 -> 512, 3 ResNet blocks, upsample 2x)
  ├── ResnetBlock2D(512 -> 512) x3
  └── Upsample2D(512)
  |
Up Block 1: (512 -> 512, 3 ResNet blocks, upsample 2x)
  ├── ResnetBlock2D(512 -> 512) x3
  └── Upsample2D(512)
  |
Up Block 2: (512 -> 256, 3 ResNet blocks, upsample 2x)
  ├── ResnetBlock2D(512 -> 256), ResnetBlock2D(256 -> 256) x2
  └── Upsample2D(256)
  |
Up Block 3: (256 -> 128, 3 ResNet blocks, NO upsample)
  ├── ResnetBlock2D(256 -> 128), ResnetBlock2D(128 -> 128) x2
  └── (no upsample — final resolution)
  |
conv_norm_out: GroupNorm(32, 128)
SiLU activation
conv_out: Conv2d(128, 3, kernel=3, padding=1)
  |
Output: [B, H, W, 3] pixel data
```

### Sub-components to implement

Each as an `MLXNN.Module` subclass in `SDXLVAEModel.swift`:

1. **`ResnetBlock2D`** — Residual block with GroupNorm + SiLU + Conv2d
   - Parameters: `norm1`, `conv1`, `norm2`, `conv2`, optional `conv_shortcut` (when channels change)
   - GroupNorm with 32 groups
   - Forward: `x + conv2(silu(norm2(conv1(silu(norm1(x))))))`
   - Skip connection with optional 1x1 conv for channel matching

2. **`AttentionBlock`** — Self-attention (single head for VAE)
   - Parameters: `group_norm`, `query`, `key`, `value`, `proj_attn`
   - Reshapes spatial dims to sequence for attention, then back

3. **`Upsample2D`** — 2x nearest-neighbor upsampling + Conv2d
   - Parameters: `conv` (3x3, same channels)
   - Forward: nearest_upsample(x, 2) -> conv

4. **`VAEMidBlock`** — Mid-block: ResNet + Attention + ResNet

5. **`VAEUpBlock`** — Up-block: N ResNet blocks + optional Upsample2D

6. **`SDXLVAEDecoderModel`** — Full decoder stack
   - `post_quant_conv` + mid_block + 4 up_blocks + norm + SiLU + conv_out

### Weight key mapping (INF-7)

```
post_quant_conv.{weight,bias}                    -> post_quant_conv.{weight,bias}
decoder.mid_block.attentions.0.{component}       -> mid_block.attention.{component}
decoder.mid_block.resnets.{i}.{component}        -> mid_block.resnets.{i}.{component}
decoder.up_blocks.{i}.resnets.{j}.{component}    -> up_blocks.{i}.resnets.{j}.{component}
decoder.up_blocks.{i}.upsamplers.0.conv.{w,b}    -> up_blocks.{i}.upsample.conv.{w,b}
decoder.conv_norm_out.{weight,bias}              -> conv_norm_out.{weight,bias}
decoder.conv_out.{weight,bias}                   -> conv_out.{weight,bias}
```

~130 keys total across all blocks.

**Important**: SDXL VAE weights from diffusers use NCHW tensor layout for convolutions. MLX Conv2d expects NHWC by default. The `tensorTransform` must transpose conv weight tensors from `[out, in, kH, kW]` to `[out, kH, kW, in]`.

### `apply(weights:)` implementation

Must load parameters into all sub-modules. Use `MLXNN.Module.update(parameters:)` with the nested dictionary structure matching the module hierarchy.

### Exit criteria

- [ ] `decode(latents)` with loaded weights produces `[B, H, W, 3]` pixel data with values in a plausible range (roughly 0-1 after clamping)
- [ ] Output resolution is 8x the latent spatial dimensions
- [ ] Scaling factor (1/0.13025) is applied internally before the forward pass
- [ ] GroupNorm with 32 groups operates correctly on all channel dimensions
- [ ] Memory estimate matches actual loaded weight size within 10%
- [ ] Unit test: synthetic weights -> correct output shape, non-constant values
- [ ] Integration test (gated): real SDXL VAE fp16 weights -> known latent input -> PSNR against reference image

---

## INF-4: LoRA Integration in DiffusionPipeline

**File**: `Sources/Tuberia/Pipeline/DiffusionPipeline.swift`
**Priority**: P1 (functional pipeline works without LoRA; this enables fine-tuned generation)

### Current state

LoRA adapter weights are loaded but then discarded:
```swift
// Line ~247-258: loads adapter weights
let loraAdapterWeights = try await LoRALoader.loadAdapterWeights(...)

// Line ~459: discards them
_ = loraAdapterWeights
```

### Required changes

Replace the load-and-discard pattern with actual apply/unapply:

```swift
// After loading adapter weights (line ~258), APPLY them:
if let adapterWeights = loraAdapterWeights, let loraConfig = request.loRA {
    // Get current backbone weights (need a way to access them)
    // Apply LoRA: merges adapter into backbone weights
    let mergedWeights = LoRALoader.apply(
        adapterWeights: adapterWeights,
        to: backboneWeights,
        scale: loraConfig.scale
    )
    try backbone.apply(weights: mergedWeights)
}

// ... generation loop ...

// After generation (line ~456-459), UNAPPLY:
if let adapterWeights = loraAdapterWeights, let loraConfig = request.loRA {
    let restoredWeights = LoRALoader.unapply(
        adapterWeights: adapterWeights,
        from: currentBackboneWeights,
        scale: loraConfig.scale
    )
    try backbone.apply(weights: restoredWeights)
}
```

### Design consideration: weight access

`LoRALoader.apply()` needs the backbone's current weights as `ModuleParameters`. The backbone must expose a way to extract its current weights. Options:

**Option A**: Add a `currentWeights` property to `WeightedSegment` protocol. Clean but requires protocol change.

**Option B**: Cache the loaded weights in `DiffusionPipeline` during `loadModels()`. No protocol change but duplicates memory.

**Option C**: `LoRALoader.apply()` works directly on the `MLXNN.Module` parameter tree (using `Module.parameters()`). Tightest integration with MLX but couples LoRA to MLXNN internals.

**Recommendation**: Option A — `WeightedSegment` should expose `currentWeights` as an optional read-only property. The pipeline can then access weights without caching.

### Exit criteria

- [ ] LoRA adapter weights are applied to backbone before the denoising loop
- [ ] LoRA adapter weights are removed (base weights restored) after generation
- [ ] `loraConfig.scale` correctly modulates the adapter effect
- [ ] LoRA with `scale=0.0` produces identical output to no-LoRA generation
- [ ] Single active LoRA constraint is enforced
- [ ] Unit test: mock backbone + synthetic LoRA -> weights change during generation, restore after

---

## INF-5: CGImage-to-MLXArray Conversion for img2img

**File**: `Sources/Tuberia/Pipeline/DiffusionPipeline.swift`
**Priority**: P2 (img2img is secondary to text-to-image)

### Current state

```swift
// Line ~346: placeholder zeros instead of actual pixel data
let referencePixels = MLXArray.zeros([1, request.height, request.width, 3])
```

### Required implementation

Convert `CGImage` to `MLXArray` with shape `[1, H, W, 3]` and values in `[0, 1]`:

```swift
private func cgImageToMLXArray(_ image: CGImage, height: Int, width: Int) -> MLXArray {
    // 1. Create CGContext with RGB colorspace, 8-bit per channel
    // 2. Draw the CGImage scaled to (width, height)
    // 3. Extract raw pixel bytes (RGBA or RGB)
    // 4. Convert UInt8 -> Float32, normalize to [0, 1]
    // 5. Reshape to [1, height, width, 3] (drop alpha if RGBA)
    // 6. Return as MLXArray
}
```

### Exit criteria

- [ ] `CGImage` with known pixel values converts to correct `MLXArray` values
- [ ] Output shape is `[1, H, W, 3]` regardless of input image dimensions (resized)
- [ ] Values are normalized to `[0, 1]` range
- [ ] Alpha channel is dropped (RGB only)
- [ ] Works with both RGB and RGBA source images
- [ ] Unit test: synthetic CGImage -> known MLXArray values

---

## INF-6: T5-XXL Weight Key Mapping

Covered as part of INF-1. See the key mapping table in that section.

**Priority**: P0 (required by INF-1)

### Exit criteria

- [ ] All ~580 encoder keys map correctly
- [ ] Decoder keys (`decoder.*`, `lm_head.*`) are filtered out (return `nil`)
- [ ] `shared.weight` maps to embedding table
- [ ] Relative position bias from layer 0 is correctly routed
- [ ] Unit test: synthetic key list -> expected mapped keys with no misses

---

## INF-7: SDXL VAE Weight Key Mapping

Covered as part of INF-3. See the key mapping table in that section.

**Priority**: P0 (required by INF-3)

### Additional requirement: tensor transforms

SDXL VAE convolution weights use NCHW layout in diffusers safetensors. MLX Conv2d expects weight shape `[out_channels, kH, kW, in_channels]`. The `tensorTransform` must:

```swift
{ key, tensor in
    // Transpose conv weights from [out, in, kH, kW] to [out, kH, kW, in]
    if key.contains("conv") && tensor.ndim == 4 {
        return tensor.transposed(0, 2, 3, 1)
    }
    return tensor
}
```

### Exit criteria

- [ ] All ~130 decoder keys map correctly
- [ ] Encoder keys (`encoder.*`, `quant_conv.*`) are filtered out
- [ ] Conv weight tensors are transposed from NCHW to NHWC
- [ ] GroupNorm weight/bias shapes are preserved (1D)
- [ ] Unit test: synthetic key list + tensor shapes -> correct mapped keys and transposed shapes

---

## INF-8: FlowMatchEulerScheduler Robustness

**File**: `Sources/TuberiaCatalog/Schedulers/FlowMatchEulerScheduler.swift`
**Priority**: P2 (current implementation works for happy path)

### Issues

1. **Line 64**: When `currentPlan` is nil, falls back to `sample + output * 0.01` — arbitrary constant
2. **Line 69-70**: When timestep not found in plan, returns `sample` unchanged — silently drops a step

### Required changes

1. When `currentPlan` is nil, throw a clear error instead of silently using a hardcoded dt
2. When timestep not found, find the nearest timestep index or throw with diagnostic info

### Exit criteria

- [ ] `step()` without prior `configure()` throws a descriptive error
- [ ] `step()` with an out-of-plan timestep either snaps to nearest or throws with the expected vs. available timesteps
- [ ] Test: step without configure -> error
- [ ] Test: step with invalid timestep -> predictable behavior

---

## Implementation Order

```
Phase 1: Neural architectures (P0, sequential)
  INF-1 + INF-2 + INF-6  (T5-XXL encoder + tokenizer + key mapping)
  INF-3 + INF-7           (SDXL VAE decoder + key mapping + tensor transform)

Phase 2: Pipeline wiring (P1, after Phase 1)
  INF-4                   (LoRA apply/unapply in DiffusionPipeline)

Phase 3: Secondary features (P2, independent)
  INF-5                   (CGImage conversion for img2img)
  INF-8                   (FlowMatchEuler robustness)
```

Phases 1a (T5-XXL) and 1b (SDXL VAE) can run in parallel — they share no code. Phase 2 depends on Phase 1 completing (LoRA needs a real backbone with weights). Phase 3 items are independent of everything else.

---

## Testing Strategy

All inference tests that require real weights are gated behind `#if INTEGRATION_TESTS`. Unit tests use synthetic weights and verify:

1. **Shape correctness** — output dimensions match specification
2. **Non-triviality** — outputs are not all-zeros, all-constant, or all-NaN
3. **Determinism** — same seed + same input = same output
4. **Key mapping completeness** — every expected key is mapped, no orphans

Integration tests (when real weights are available) additionally verify:

5. **PSNR against reference** — output matches a known-good reference within tolerance
6. **Round-trip consistency** — encode -> decode produces plausible reconstruction

---

## Weight Availability

Real inference requires converted weights on HuggingFace:

| Component | HuggingFace Repo | Status |
|-----------|-----------------|--------|
| T5-XXL int4 | `intrusive-memory/t5-xxl-int4-mlx` | Not yet created |
| SDXL VAE fp16 | `intrusive-memory/sdxl-vae-fp16-mlx` | Not yet created |

Weight conversion is an external dependency. Unit tests must work without real weights. Integration tests gate on weight availability.
