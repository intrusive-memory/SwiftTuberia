---
feature_name: OPERATION AWAKENING FORGE
starting_point_commit: a5dfd5c6c6c921be4e36220c2a72785925561717
mission_branch: mission/awakening-forge/1
iteration: 1
---

# EXECUTION_PLAN.md — SwiftTuberia Inference Implementation

**Source**: `requirements/INFERENCE.md`
**Date**: 2026-03-26
**Predecessor**: [Execution Plan 01](docs/complete/swifttuberia-pipeline-system-01-execution-plan.md) (pipeline plumbing — complete)
**Status**: REFINED — ready for execution

---

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

---

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|--------------|
| T5-XXL Encoder | Sources/TuberiaCatalog/Encoders/ | 3 | 1 | none |
| SDXL VAE Decoder | Sources/TuberiaCatalog/Decoders/ | 3 | 1 | none |
| Pipeline LoRA Integration | Sources/Tuberia/ | 1 | 2 | T5-XXL Encoder, SDXL VAE Decoder |
| Secondary Features | Sources/ | 2 | 1 | none |

**Layer 1** work units (T5-XXL, SDXL VAE, Secondary Features) can execute in parallel.
**Layer 2** work units (Pipeline LoRA) depend on all Layer 1 work units completing.

---

## Parallelism Structure

**Critical Path**: T5 S1 → T5 S2 → T5 S3 → LoRA S1 (4 sorties)

**Parallel Execution Groups**:
- **Group 1** (all independent — launch simultaneously):
  - T5-XXL Encoder: Sortie 1 (Agent 1)
  - SDXL VAE Decoder: Sortie 1 (Agent 2)
  - Secondary Features: Sortie 1 — CGImage (Agent 3)
  - Secondary Features: Sortie 2 — FlowMatch (Agent 4)
- **Group 2** (after Group 1 — parallel across work units):
  - T5-XXL Encoder: Sortie 2 (Agent 1)
  - SDXL VAE Decoder: Sortie 2 (Agent 2)
- **Group 3** (after Group 2 — parallel across work units):
  - T5-XXL Encoder: Sortie 3 (Agent 1) — **SUPERVISING AGENT** (modifies Package.swift)
  - SDXL VAE Decoder: Sortie 3 (Agent 2)
- **Group 4** (after Group 3 — sequential):
  - Pipeline LoRA Integration: Sortie 1 (Agent 1)

**Agent Constraints**:
- **Supervising agent**: T5 S3 (modifies shared Package.swift)
- **All sorties require builds**: Parallel builds are safe in Groups 1-2 because work units touch independent source files

---

## Work Unit: T5-XXL Encoder

### Sortie 1: T5 Transformer Architecture Modules

**Priority**: 14.0 — Highest dependency depth (blocks 3 downstream sorties) + foundational architecture

**Goal**: Build all 5 MLXNN.Module subclasses for the T5-XXL encoder-only transformer in a new file.

**Estimated turns**: 17/50 (34%)

**Entry criteria**:
- [ ] First sortie — no prerequisites
- [ ] `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` exists with placeholder implementation

**Tasks**:
1. Read `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` to understand existing placeholder structure
2. Read `requirements/INFERENCE.md` § INF-1 for full architecture specification
3. Create `Sources/TuberiaCatalog/Encoders/T5TransformerEncoder.swift`
4. Implement `T5RMSNorm` module — RMS layer normalization (no bias, no mean subtraction, eps=1e-6). Parameters: `weight` [hidden_dim]. Forward: `x * rsqrt(mean(x^2) + eps) * weight`
5. Implement `T5Attention` module — Multi-head self-attention (64 heads, head_dim=64) with relative position bias (32 buckets, bidirectional). Parameters: `q`, `k`, `v`, `o` projection matrices [4096, 4096], no bias. `relative_attention_bias` [num_heads, num_buckets] only on layer 0, shared across layers
6. Implement `T5GatedFFN` module — GeGLU gated feed-forward. Parameters: `wi_0` [4096, 10240], `wi_1` [4096, 10240], `wo` [10240, 4096], no bias. Forward: `wo(gelu(wi_0(x)) * wi_1(x))`
7. Implement `T5EncoderBlock` module — Single transformer block: pre-attn RMSNorm → T5Attention → residual → pre-FFN RMSNorm → T5GatedFFN → residual
8. Implement `T5TransformerEncoder` module — Full encoder stack: token embedding (32128, 4096) → 24 encoder blocks → final RMSNorm → output [B, seq_len, 4096]

**Exit criteria**:
- [ ] `T5TransformerEncoder.swift` compiles with no errors: `xcodebuild build -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64'`
- [ ] All 5 module classes (`T5RMSNorm`, `T5Attention`, `T5GatedFFN`, `T5EncoderBlock`, `T5TransformerEncoder`) are defined as `MLXNN.Module` subclasses (verify: `grep -c 'class T5.*: .*Module' Sources/TuberiaCatalog/Encoders/T5TransformerEncoder.swift` returns 5)
- [ ] `T5TransformerEncoder` has a callable forward method accepting token IDs `MLXArray` [B, seq_len] and returning embeddings `MLXArray` [B, seq_len, 4096] (verify: `grep 'func callAsFunction\|func forward' Sources/TuberiaCatalog/Encoders/T5TransformerEncoder.swift`)
- [ ] Relative position bias is shared across all layers — single `relative_attention_bias` property on the encoder, passed into attention layers (verify: `grep -c 'relative_attention_bias\|relative_position_bias' Sources/TuberiaCatalog/Encoders/T5TransformerEncoder.swift` shows 1-2 declaration sites, not 24)

---

### Sortie 2: T5 Key Mapping and Weight Loading

**Priority**: 9.0 — Blocks 2 downstream sorties

**Goal**: Implement the ~580-key mapping for T5 safetensors and wire `apply(weights:)` to load parameters into the transformer modules.

**Estimated turns**: 20/50 (40%)

**Entry criteria**:
- [ ] Sortie 1 exit criteria met — `T5TransformerEncoder` module hierarchy exists and compiles
- [ ] `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` exists with identity-passthrough key mapping

**Tasks**:
1. Read `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` to understand current `keyMapping` implementation
2. Read `Sources/TuberiaCatalog/Encoders/T5TransformerEncoder.swift` to understand module property names
3. Read `requirements/INFERENCE.md` § INF-1 → Weight key mapping table
4. Replace the identity-passthrough `keyMapping` in `T5XXLEncoder.swift` with the real mapping function that translates safetensors keys (e.g., `encoder.block.{i}.layer.0.SelfAttention.q.weight` → `blocks.{i}.attention.q.weight`)
5. Filter decoder keys: `decoder.*` and `lm_head.*` → return `nil` (skip)
6. Map `shared.weight` → `embedding.weight`
7. Route `encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight` → `relative_position_bias.weight`
8. Map `encoder.final_layer_norm.weight` → `final_norm.weight`
9. Implement `apply(weights:)` in `T5XXLEncoder.swift` to load `ModuleParameters` into the `T5TransformerEncoder` module tree using `MLXNN.Module.update(parameters:)` or manual parameter assignment
10. Write unit tests in `Tests/TuberiaCatalogTests/` verifying: (a) all ~580 expected key patterns map correctly, (b) decoder/lm_head keys return nil, (c) `shared.weight` maps to embedding, (d) relative position bias is routed from layer 0

**Exit criteria**:
- [ ] Key mapping function produces correct output for all 24 layers × 11 keys/layer + embedding + final norm + relative position bias (verified by unit tests)
- [ ] `decoder.*` and `lm_head.*` keys return `nil` (verified by unit tests)
- [ ] `apply(weights:)` calls through to module parameter loading — compiles and unit test with synthetic `ModuleParameters` does not crash
- [ ] Unit tests pass: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaCatalogTests`

---

### Sortie 3: Tokenizer Integration and Encoder Wiring

**Priority**: 7.25 — External dependency risk (swift-transformers), blocks LoRA

**Goal**: Load the T5 tokenizer via swift-transformers, wire tokenization into the encode path, and connect the real transformer forward pass.

**Estimated turns**: 23/50 (46%)

**Entry criteria**:
- [ ] Sortie 2 exit criteria met — key mapping and apply(weights:) are functional
- [ ] `T5TransformerEncoder` module compiles and accepts token IDs as input

**Tasks**:
1. Read `Package.swift` to understand current dependency structure
2. Read `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` to understand current `encode()` method
3. Add `swift-transformers` package dependency to `Package.swift` (url: `https://github.com/huggingface/swift-transformers`) and add the `Tokenizers` product to the `TuberiaCatalog` target dependencies
4. Add a `loadTokenizer()` async method to `T5XXLEncoder` that loads the tokenizer from Acervo component directory using `AutoTokenizer.from(modelFolder:)`. Store as `private var tokenizer: (any Tokenizer)?`
5. Update `T5XXLEncoder.encode()` to: (a) tokenize input text using the loaded tokenizer, (b) clamp to `maxSequenceLength`, (c) pad with pad token (ID 0) if shorter, (d) generate attention mask (1 for real tokens, 0 for padding), (e) run the `T5TransformerEncoder` forward pass when loaded, (f) fall back to placeholder when weights not loaded
6. Wire `loadTokenizer()` into the pipeline load phase — add call in `DiffusionPipeline.loadModels()` or via protocol extension on `TextEncoder` (recommendation: separate `loadTokenizer()` called during pipeline load, per requirements/INFERENCE.md § INF-2)
7. Write unit tests verifying: (a) tokenization produces plausible token count for "a photo of a cat" (4-7 tokens), (b) padding generates correct attention mask, (c) `maxSequenceLength` truncation works, (d) empty string tokenizes to single EOS/pad token, (e) encode output shape is [1, seqLen, 4096] with non-zero, non-constant values (using synthetic weights)

**Exit criteria**:
- [ ] `swift-transformers` dependency added to Package.swift and TuberiaCatalog target (verify: `grep 'swift-transformers' Package.swift`)
- [ ] `T5XXLEncoder` has `loadTokenizer()` async method (verify: `grep 'func loadTokenizer' Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift`)
- [ ] `encode()` uses real tokenizer when available, falls back to placeholder when not (verified by unit tests)
- [ ] `encode()` with loaded transformer produces `[1, seqLen, 4096]` non-zero embeddings with synthetic weights (verified by unit tests)
- [ ] Attention mask correctly marks real tokens as 1 and padding as 0 (verified by unit tests)
- [ ] Project builds: `xcodebuild build -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64'`
- [ ] Unit tests pass: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaCatalogTests`

---

## Work Unit: SDXL VAE Decoder

### Sortie 1: VAE Decoder Architecture Modules

**Priority**: 14.0 — Highest dependency depth (blocks 3 downstream sorties) + foundational architecture

**Goal**: Build all 6 MLXNN.Module subclasses for the SDXL VAE decoder in a new file.

**Estimated turns**: 18/50 (36%)

**Entry criteria**:
- [ ] First sortie — no prerequisites
- [ ] `Sources/TuberiaCatalog/Decoders/SDXLVAEDecoder.swift` exists with placeholder implementation

**Tasks**:
1. Read `Sources/TuberiaCatalog/Decoders/SDXLVAEDecoder.swift` to understand existing placeholder structure
2. Read `requirements/INFERENCE.md` § INF-3 for full architecture specification
3. Create `Sources/TuberiaCatalog/Decoders/SDXLVAEModel.swift`
4. Implement `ResnetBlock2D` module — GroupNorm(32) + SiLU + Conv2d residual block with optional `conv_shortcut` (1x1 conv) for channel dimension changes. Forward: `x + conv2(silu(norm2(conv1(silu(norm1(x))))))`
5. Implement `AttentionBlock` module — Single-head self-attention for VAE. Parameters: `group_norm`, `query`, `key`, `value`, `proj_attn`. Reshapes spatial dims [B, H, W, C] → [B, H*W, C] for attention, then back
6. Implement `Upsample2D` module — 2x nearest-neighbor upsampling + Conv2d(3x3, same channels). Forward: `conv(nearest_upsample(x, 2))`
7. Implement `VAEMidBlock` module — ResnetBlock2D + AttentionBlock + ResnetBlock2D
8. Implement `VAEUpBlock` module — N ResnetBlock2D blocks + optional Upsample2D. Parameterized by input/output channels, number of ResNet blocks, and whether upsample is present
9. Implement `SDXLVAEDecoderModel` module — Full decoder stack: post_quant_conv(4→4, 1x1) → VAEMidBlock → 4 VAEUpBlocks (512→512→256→128 channels, blocks 0-2 upsample, block 3 no upsample) → GroupNorm(32, 128) → SiLU → Conv2d(128→3, 3x3, padding=1) → output [B, H, W, 3]

**Exit criteria**:
- [ ] `SDXLVAEModel.swift` compiles with no errors: `xcodebuild build -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64'`
- [ ] All 6 module classes (`ResnetBlock2D`, `AttentionBlock`, `Upsample2D`, `VAEMidBlock`, `VAEUpBlock`, `SDXLVAEDecoderModel`) are defined as `MLXNN.Module` subclasses (verify: `grep -c 'class.*: .*Module' Sources/TuberiaCatalog/Decoders/SDXLVAEModel.swift` returns 6)
- [ ] `SDXLVAEDecoderModel` has a callable forward method accepting latents `MLXArray` [B, H/8, W/8, 4] and returning pixels `MLXArray` [B, H, W, 3] (verify: `grep 'func callAsFunction\|func forward' Sources/TuberiaCatalog/Decoders/SDXLVAEModel.swift`)
- [ ] Channel progression verified: `grep -o '[0-9]\+' Sources/TuberiaCatalog/Decoders/SDXLVAEModel.swift | sort -u` includes 4, 128, 256, 512

---

### Sortie 2: VAE Key Mapping, Tensor Transforms, and Weight Loading

**Priority**: 9.0 — Blocks 2 downstream sorties

**Goal**: Implement the ~130-key mapping for SDXL VAE safetensors with NCHW→NHWC tensor transposition, and wire `apply(weights:)`.

**Estimated turns**: 21/50 (42%)

**Entry criteria**:
- [ ] Sortie 1 exit criteria met — `SDXLVAEDecoderModel` module hierarchy exists and compiles
- [ ] `Sources/TuberiaCatalog/Decoders/SDXLVAEDecoder.swift` exists with partial key mapping

**Tasks**:
1. Read `Sources/TuberiaCatalog/Decoders/SDXLVAEDecoder.swift` to understand current `keyMapping` implementation
2. Read `Sources/TuberiaCatalog/Decoders/SDXLVAEModel.swift` to understand module property names
3. Read `requirements/INFERENCE.md` § INF-3 and INF-7 for key mapping table and tensor transform requirements
4. Replace the key mapping in `SDXLVAEDecoder.swift` with the real mapping function that translates safetensors keys (e.g., `decoder.up_blocks.{i}.resnets.{j}.{component}` → `up_blocks.{i}.resnets.{j}.{component}`)
5. Filter encoder keys: `encoder.*` and `quant_conv.*` → return `nil` (skip)
6. Implement `tensorTransform` on `SDXLVAEDecoder` that transposes Conv2d weights from NCHW `[out, in, kH, kW]` to NHWC `[out, kH, kW, in]` for 4D tensors whose key contains "conv"
7. Implement `apply(weights:)` in `SDXLVAEDecoder.swift` to load `ModuleParameters` into the `SDXLVAEDecoderModel` module tree
8. Write unit tests in `Tests/TuberiaCatalogTests/` verifying: (a) all ~130 expected key patterns map correctly, (b) encoder/quant_conv keys return nil, (c) tensor transform transposes 4D conv weights correctly, (d) GroupNorm weight/bias shapes (1D) are preserved

**Exit criteria**:
- [ ] Key mapping function produces correct output for all decoder keys: mid block, 4 up blocks, post_quant_conv, conv_norm_out, conv_out (verified by unit tests)
- [ ] `encoder.*` and `quant_conv.*` keys return `nil` (verified by unit tests)
- [ ] `tensorTransform` transposes `[out, in, kH, kW]` → `[out, kH, kW, in]` for conv weights (verified by unit tests)
- [ ] `apply(weights:)` calls through to module parameter loading — compiles and unit test with synthetic `ModuleParameters` does not crash
- [ ] Unit tests pass: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaCatalogTests`

---

### Sortie 3: VAE Decoder Integration and Forward Pass Wiring

**Priority**: 5.0 — Terminal sortie for VAE work unit

**Goal**: Wire the real `SDXLVAEDecoderModel` into `SDXLVAEDecoder`, replacing the placeholder forward pass.

**Estimated turns**: 17/50 (34%)

**Entry criteria**:
- [ ] Sortie 2 exit criteria met — key mapping, tensor transform, and apply(weights:) are functional
- [ ] `SDXLVAEDecoderModel` accepts latents and returns pixel data

**Tasks**:
1. Read `Sources/TuberiaCatalog/Decoders/SDXLVAEDecoder.swift` to understand current `decode()` and `apply(weights:)` methods
2. Read `Sources/TuberiaCatalog/Decoders/SDXLVAEModel.swift` to understand `SDXLVAEDecoderModel` forward method signature
3. Add a `private var model: SDXLVAEDecoderModel?` property to `SDXLVAEDecoder`
4. Instantiate `SDXLVAEDecoderModel` in `apply(weights:)` (or lazily on first decode), then load parameters into it
5. Update `decode()` to: (a) apply scaling factor `latents * (1.0 / scalingFactor)`, (b) run `model.forward(scaledLatents)` when loaded, (c) fall back to `placeholderForwardPass()` when model is nil
6. Update `unload()` to nil out the model and release memory
7. Write unit tests verifying: (a) `decode(latents)` with synthetic weights produces `[B, H, W, 3]` output, (b) output spatial dimensions are 8× latent spatial dimensions, (c) values are in a plausible range after clamping, (d) unloaded decoder falls back to placeholder

**Exit criteria**:
- [ ] `SDXLVAEDecoder.decode()` uses real `SDXLVAEDecoderModel` when weights are loaded (verify: `grep 'model.*forward\|model.*callAsFunction' Sources/TuberiaCatalog/Decoders/SDXLVAEDecoder.swift`)
- [ ] Unloaded decoder still produces correctly-shaped placeholder output (verified by unit tests)
- [ ] Output shape is `[B, H, W, 3]` with spatial dims 8× input latent spatial dims (verified by unit tests)
- [ ] Scaling factor `1/0.13025` is applied before forward pass (verify: `grep '0.13025\|scalingFactor' Sources/TuberiaCatalog/Decoders/SDXLVAEDecoder.swift`)
- [ ] Project builds: `xcodebuild build -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64'`
- [ ] Unit tests pass: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaCatalogTests`

---

## Work Unit: Pipeline LoRA Integration

### Sortie 1: LoRA Apply/Unapply in DiffusionPipeline

**Priority**: 3.5 — Layer 2 (depends on all Layer 1 work units completing)

**Goal**: Replace the LoRA load-and-discard pattern with actual weight merging before the denoising loop and restoration after generation.

**Estimated turns**: 31/50 (62%)

**Entry criteria**:
- [ ] T5-XXL Encoder work unit completed (all 3 sorties)
- [ ] SDXL VAE Decoder work unit completed (all 3 sorties)
- [ ] `LoRALoader.apply()` and `LoRALoader.unapply()` already exist in `Sources/Tuberia/Pipeline/LoRALoader.swift` (verify: `grep -c 'func apply\|func unapply' Sources/Tuberia/Pipeline/LoRALoader.swift` returns 2)
- [ ] `DiffusionPipeline.swift` has the load-and-discard pattern at ~lines 247-258 and ~459

**Tasks**:
1. Read `Sources/Tuberia/Protocols/WeightedSegment.swift` to understand the current protocol
2. Read `Sources/Tuberia/Pipeline/DiffusionPipeline.swift` around lines 244-260 and 456-460 to understand the load-and-discard pattern
3. Read `Sources/Tuberia/Pipeline/LoRALoader.swift` to understand `apply()` and `unapply()` signatures
4. Find all `WeightedSegment` conformers in the codebase (grep for `: WeightedSegment` or `WeightedSegment` conformance)
5. Add a `var currentWeights: ModuleParameters? { get }` property requirement to the `WeightedSegment` protocol in `Sources/Tuberia/Protocols/WeightedSegment.swift`
6. Implement `currentWeights` in ALL `WeightedSegment` conformers found in step 4 (return the cached `ModuleParameters` from the last `apply(weights:)` call, or nil if unloaded)
7. In `DiffusionPipeline.generate()`, after loading LoRA adapter weights (~line 258), apply them to the backbone: `let mergedWeights = LoRALoader.apply(adapterWeights: adapterWeights, to: backbone.currentWeights!, scale: loraConfig.scale)` then `try backbone.apply(weights: mergedWeights)`
8. In `DiffusionPipeline.generate()`, after the generation loop (~line 459), unapply LoRA: `let restoredWeights = LoRALoader.unapply(adapterWeights: adapterWeights, from: backbone.currentWeights!, scale: loraConfig.scale)` then `try backbone.apply(weights: restoredWeights)`
9. Ensure single active LoRA constraint: verify only one LoRA config is processed per generation call
10. Write unit tests in `Tests/TuberiaTests/`: (a) mock backbone + synthetic LoRA → weights change during generation window, (b) weights are restored to original values after generation, (c) LoRA with scale=0.0 produces no weight change

**Exit criteria**:
- [ ] `WeightedSegment` protocol has `currentWeights` property (verify: `grep 'currentWeights' Sources/Tuberia/Protocols/WeightedSegment.swift`)
- [ ] All `WeightedSegment` conformers implement `currentWeights` — build succeeds with no missing protocol requirements
- [ ] `DiffusionPipeline.generate()` applies LoRA to backbone before denoising loop (verify: `grep 'LoRALoader.apply' Sources/Tuberia/Pipeline/DiffusionPipeline.swift`)
- [ ] `DiffusionPipeline.generate()` restores base weights after generation (verify: `grep 'LoRALoader.unapply' Sources/Tuberia/Pipeline/DiffusionPipeline.swift`)
- [ ] `loraConfig.scale` correctly modulates the adapter effect — scale=0.0 → no weight change (verified by unit tests)
- [ ] Project builds: `xcodebuild build -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64'`
- [ ] Unit tests pass: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaTests`

---

## Work Unit: Secondary Features

### Sortie 1: CGImage-to-MLXArray Conversion for img2img

**Priority**: 3.0 — Independent, no downstream dependencies

**Goal**: Replace the placeholder zeros with real CGImage pixel extraction and conversion to MLXArray.

**Estimated turns**: 18/50 (36%)

**Entry criteria**:
- [ ] First sortie — no prerequisites
- [ ] `DiffusionPipeline.swift` has the placeholder `MLXArray.zeros([1, request.height, request.width, 3])` at ~line 346

**Tasks**:
1. Read `Sources/Tuberia/Pipeline/DiffusionPipeline.swift` around line 346 to understand the img2img placeholder
2. Implement a `private func cgImageToMLXArray(_ image: CGImage, height: Int, width: Int) -> MLXArray` method in `DiffusionPipeline.swift`
3. Create a `CGContext` with RGB color space, 8-bit per channel, draw the `CGImage` scaled to `(width, height)`
4. Extract raw pixel bytes (handle both RGBA and RGB by dropping alpha channel)
5. Convert `UInt8` → `Float32`, normalize to `[0, 1]`, reshape to `[1, height, width, 3]`
6. Replace the `MLXArray.zeros` placeholder in `generate()` with a call to `cgImageToMLXArray(image, height:, width:)`
7. Write unit tests in `Tests/TuberiaTests/`: (a) synthetic CGImage with known pixel values → correct MLXArray values, (b) output shape is `[1, H, W, 3]`, (c) values are in `[0, 1]`, (d) alpha channel is dropped, (e) works with both RGB and RGBA source images

**Exit criteria**:
- [ ] `cgImageToMLXArray()` exists in DiffusionPipeline.swift (verify: `grep 'func cgImageToMLXArray' Sources/Tuberia/Pipeline/DiffusionPipeline.swift`)
- [ ] Values are normalized to `[0, 1]` range (verified by unit tests)
- [ ] Alpha channel is correctly dropped — RGB only output (verified by unit tests)
- [ ] Both RGB and RGBA source images are handled (verified by unit tests)
- [ ] Placeholder `MLXArray.zeros` is replaced with real conversion (verify: `grep -c 'MLXArray.zeros.*request.height.*request.width.*3' Sources/Tuberia/Pipeline/DiffusionPipeline.swift` returns 0)
- [ ] Project builds: `xcodebuild build -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64'`
- [ ] Unit tests pass: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaTests`

---

### Sortie 2: FlowMatchEulerScheduler Robustness

**Priority**: 1.5 — Independent, lowest risk

**Goal**: Replace silent fallbacks in `FlowMatchEulerScheduler.step()` with descriptive errors or nearest-timestep snapping.

**Estimated turns**: 15/50 (30%)

**Entry criteria**:
- [ ] First sortie — no prerequisites
- [ ] `Sources/TuberiaCatalog/Schedulers/FlowMatchEulerScheduler.swift` exists with fallback behavior at lines 64 and 69-70

**Tasks**:
1. Read `Sources/TuberiaCatalog/Schedulers/FlowMatchEulerScheduler.swift` to understand current fallback behavior
2. Replace the fallback at line 64 (`return sample + output * 0.01` when `currentPlan` is nil) with a thrown `PipelineError` that describes the issue: step() called without prior configure()
3. Replace the silent return at lines 69-70 (`return sample` when timestep not found in plan) with either: (a) snap to nearest timestep index in the plan, or (b) throw a `PipelineError` listing the expected vs. available timesteps. Agent chooses the approach — both are acceptable
4. Write test in `Tests/TuberiaCatalogTests/`: calling `step()` without prior `configure()` throws a descriptive error
5. Write test in `Tests/TuberiaCatalogTests/`: calling `step()` with an out-of-plan timestep produces predictable behavior (error or snap, matching chosen approach)

**Exit criteria**:
- [ ] `step()` without prior `configure()` throws a descriptive `PipelineError` (verify: `grep -c 'throw.*PipelineError\|throw.*configure' Sources/TuberiaCatalog/Schedulers/FlowMatchEulerScheduler.swift` ≥ 1)
- [ ] `step()` with an out-of-plan timestep either snaps to nearest or throws with expected vs. available timesteps — no silent `return sample` (verify: line 69-70 no longer returns unchanged sample)
- [ ] No silent data corruption — no arbitrary constants or silent no-ops (verify: `grep -c 'return sample + output \* 0.01' Sources/TuberiaCatalog/Schedulers/FlowMatchEulerScheduler.swift` returns 0)
- [ ] Project builds: `xcodebuild build -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64'`
- [ ] Unit tests pass: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaCatalogTests`

---

## Open Questions & Missing Documentation

No blocking issues found. All items resolved during refinement:

| Sortie | Issue Type | Resolution |
|--------|-----------|------------|
| T5 S1 | Vague criterion | Added grep-based verification for relative position bias sharing |
| CGImage S1 | Open question (method location) | Resolved to DiffusionPipeline.swift — matches existing code patterns |
| FlowMatch S2 | Alternative approaches | Both snap-to-nearest and throw-error are acceptable — exit criteria accept either |

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 4 |
| Total sorties | 9 |
| Dependency structure | 2 layers (Layer 1: T5-XXL ∥ SDXL VAE ∥ Secondary; Layer 2: LoRA Integration) |
| Average sortie size | 20 turns (budget: 50) |
| Critical path length | 4 sorties (T5 S1 → S2 → S3 → LoRA S1) |
| Maximum parallelism | 4 agents simultaneously (Group 1) |
| Parallel execution groups | 4 groups |
