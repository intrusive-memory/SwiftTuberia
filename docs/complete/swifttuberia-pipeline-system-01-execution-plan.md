---
feature_name: "SwiftTuberia — Pipeline System"
iteration: 1
wave: "1-2"
repository: SwiftTuberia
status: refined
---

# Execution Plan — SwiftTuberia (Refined)

## Refinement Log

**Iteration 1** — Four-pass refinement applied:

1. **Atomicity & Testability**: Sortie 1 split into 1a (Package.swift + data types) and 1b (protocol files). Sortie 4 split into 4a (PipelineProgress + PipelineError) and 4b (LoRA system). Sortie 5 split into 5a (GenerationPipeline protocol + PipelineRecipe + request/result types), 5b (DiffusionPipeline init/validation/loadModels/unloadModels), and 5c (DiffusionPipeline.generate orchestration). Sortie 6 split into 6a (mocks + contract tests) and 6b (infrastructure unit tests). Each sub-sortie now fits within a single agent context window with its required reference material.

2. **Prioritization**: Sorties scored by (impact x urgency x unblock-count). Critical path (1a -> 1b -> 2/3 -> 4a/4b -> 5a -> 5b -> 5c -> 6a/6b) is the highest priority. Catalog components (7-12) scored lower but remain wave 2 — they unblock integration testing.

3. **Parallelism**: After Sortie 1b completes, up to 4 parallel tracks are available: (A) Infrastructure: 2 -> 4b, (B) Infrastructure: 3, (C) Catalog-lightweight: 7+9+11+12, (D) Types-only: 4a + 5a. After Sortie 2 completes, 8 and 10 can join the catalog track. Maximum 4 sub-agents at any point.

4. **Open Questions**: 6 vague criteria identified and annotated inline with `[OQ-N]` markers. Summary in the Open Questions section at the end.

---

## Model Assignment Guide

| Complexity | Model | Context Budget | Sortie Examples |
|---|---|---|---|
| Low: copy-paste canonical defs, simple data transforms | haiku | ~8K tokens | 1a, 4a, 7, 12 |
| Medium: single-concern implementation with clear spec | sonnet | ~32K tokens | 1b, 2, 3, 4b, 5a, 8, 9, 10, 11, 13 |
| High: multi-concern orchestration, complex logic | opus | ~64K tokens | 5b, 5c, 6a, 6b |

---

## Current State

The repository contains a valid `Package.swift` with two targets (`Tuberia` and `TuberiaCatalog`), SwiftAcervo as the sole dependency (mlx-swift and swift-transformers still need to be added), and placeholder source/test files. No protocols, types, or infrastructure have been implemented yet. The `requirements/` directory contains complete, canonical Swift definitions for every type and protocol.

---

## Sortie 0: Reconnaissance [COMPLETE]

**Objective**: Map all requirements, validate that the canonical Swift definitions are internally consistent, and confirm the Package.swift dependency graph is correct before writing any implementation code.

**Tasks**:
- [x] Read REQUIREMENTS.md (overview), AGENTS.md, ARCHITECTURE.md
- [x] Read requirements/PROTOCOLS.md — pipe segment protocols, WeightedSegment, shape contracts
- [x] Read requirements/PIPELINE.md — DiffusionPipeline, PipelineRecipe, LoRA, errors, progress
- [x] Read requirements/CATALOG.md — T5XXL, SDXL VAE, DPM-Solver, FlowMatch, renderers, configs
- [x] Read requirements/INFRASTRUCTURE.md — WeightLoader, MemoryManager, DeviceCapability
- [x] Read requirements/TESTING.md — component tests, contract tests, integration tests, CI rules
- [x] Inventory current source files (skeleton only: placeholder comments in both targets)
- [x] Verify Package.swift targets and products match REQUIREMENTS.md

**Findings**:
- Package.swift is missing `mlx-swift` and `swift-transformers` dependencies (required by REQUIREMENTS.md).
- The `Tuberia` target needs `MLX`, `MLXNN`, `MLXRandom` product dependencies from mlx-swift.
- The `TuberiaCatalog` target needs `Tokenizers` from swift-transformers (for T5XXLEncoder).
- All canonical Swift definitions in requirements/ are internally consistent; shape contract validations reference the correct property names across protocols.
- The architecture/ directory contains duplicates of the requirements/ files (for agent consumption via ARCHITECTURE.md); the requirements/ versions are authoritative.

---

## Sortie 1a: Package.swift + Foundation Data Types (Wave 1.1a)

**Model**: haiku
**Estimated tokens**: ~4K (Package.swift edits + 4 small structs/typealiases from canonical definitions)
**Priority**: P0 (blocks everything)

**Objective**: Update Package.swift with all required dependencies and create the foundational data types that every protocol depends on.

**Files**:
- `Package.swift` — add mlx-swift, swift-transformers dependencies; wire product dependencies to targets
- `Sources/Tuberia/Types/ModuleParameters.swift` — `ModuleParameters` struct
- `Sources/Tuberia/Types/KeyMapping.swift` — `KeyMapping` typealias
- `Sources/Tuberia/Types/TensorTransform.swift` — `TensorTransform` typealias
- `Sources/Tuberia/Types/QuantizationConfig.swift` — `QuantizationConfig` enum

**Reference material for agent**:
- requirements/PROTOCOLS.md lines 186-209 (canonical Swift for ModuleParameters, KeyMapping, TensorTransform, QuantizationConfig)
- Current Package.swift

**Dependencies**: None (first sortie).

**Exit Criteria**:
- [ ] Package.swift resolves with mlx-swift, swift-transformers, SwiftAcervo
- [ ] `ModuleParameters`, `KeyMapping`, `TensorTransform`, `QuantizationConfig` are public and Sendable
- [ ] All four type files compile in the `Tuberia` target
- [ ] Types match canonical definitions verbatim from requirements/PROTOCOLS.md

---

## Sortie 1b: Pipe Segment Protocols (Wave 1.1b)

**Model**: sonnet
**Estimated tokens**: ~16K (6 protocol files with associated types, needs full PROTOCOLS.md as reference)
**Priority**: P0 (blocks all downstream sorties)

**Objective**: Implement all pipe segment protocols and the WeightedSegment lifecycle protocol from requirements/PROTOCOLS.md. These are the foundational contracts that every subsequent sortie depends on.

**Files**:
- `Sources/Tuberia/Protocols/WeightedSegment.swift` — `WeightedSegment` protocol + default extension
- `Sources/Tuberia/Protocols/TextEncoder.swift` — `TextEncoderInput`, `TextEncoderOutput`, `TextEncoder` protocol
- `Sources/Tuberia/Protocols/Scheduler.swift` — `BetaSchedule`, `PredictionType`, `SchedulerPlan`, `Scheduler` protocol with default `configure(steps:)` extension
- `Sources/Tuberia/Protocols/Backbone.swift` — `BackboneInput`, `Backbone` protocol
- `Sources/Tuberia/Protocols/Decoder.swift` — `DecoderMetadata` protocol, `ImageDecoderMetadata`, `AudioDecoderMetadata`, `DecodedOutput`, `Decoder` protocol, `BidirectionalDecoder` protocol
- `Sources/Tuberia/Protocols/Renderer.swift` — `RenderedOutput` enum, `AudioData`, `VideoFrames`, `Renderer` protocol
- Remove `Sources/Tuberia/Tuberia.swift` (placeholder)

**Reference material for agent**:
- requirements/PROTOCOLS.md (full file — canonical Swift definitions)
- Sortie 1a output files (ModuleParameters, KeyMapping, TensorTransform, QuantizationConfig)

**Dependencies**: Sortie 1a (foundation types must exist).

**Exit Criteria**:
- [ ] All protocol files compile in the `Tuberia` target
- [ ] `WeightedSegment` protocol requires `apply(weights:)`, `unload()`, `estimatedMemoryBytes`, `isLoaded`, `keyMapping`, `tensorTransform`
- [ ] Default extension provides `tensorTransform { nil }`
- [ ] `TextEncoder` protocol extends `WeightedSegment` with `associatedtype Configuration`, `init(configuration:)`, `outputEmbeddingDim`, `maxSequenceLength`, `encode(_:)`
- [ ] `Scheduler` protocol (NOT `WeightedSegment`) has `configure(steps:startTimestep:)`, `step(output:timestep:sample:)`, `addNoise(to:noise:at:)`, `reset()`
- [ ] `Backbone` protocol extends `WeightedSegment` with `expectedConditioningDim`, `outputLatentChannels`, `expectedMaxSequenceLength`, `forward(_:)`
- [ ] `Decoder` protocol extends `WeightedSegment` with `expectedInputChannels`, `scalingFactor`, `decode(_:)`
- [ ] `BidirectionalDecoder` refines `Decoder` with `encode(_:)`
- [ ] `Renderer` protocol (NOT `WeightedSegment`) has `Configuration`, `init(configuration:)`, `render(_:)`
- [ ] `RenderedOutput` enum has `.image(CGImage)`, `.audio(AudioData)`, `.video(VideoFrames)` cases
- [ ] `CGImage` reference requires `import CoreGraphics`
- [ ] Placeholder `Tuberia.swift` removed

**Notes**:
- Copy canonical Swift definitions verbatim from requirements/PROTOCOLS.md. The spec says "this code governs" over any prose.
- `Scheduler` and `Renderer` do NOT conform to `WeightedSegment` — they carry no model weights.

---

## Sortie 2: WeightLoader Infrastructure (Wave 1.2)

**Model**: sonnet
**Estimated tokens**: ~20K (WeightLoader + SafetensorsParser, needs INFRASTRUCTURE.md + PROTOCOLS.md types as reference)
**Priority**: P0 (blocks Sorties 4b, 5b, 8, 10)

**Objective**: Implement the WeightLoader service that loads safetensors files through SwiftAcervo's `withComponentAccess`, applies key remapping, tensor transforms, and quantization, then delivers clean `ModuleParameters` to pipe segments.

**Files**:
- `Sources/Tuberia/Infrastructure/WeightLoader.swift` — `WeightLoader.load(componentId:keyMapping:tensorTransform:quantization:)` static async method
- `Sources/Tuberia/Infrastructure/SafetensorsParser.swift` — internal safetensors binary format parsing (single file and sharded)

**Reference material for agent**:
- requirements/INFRASTRUCTURE.md lines 24-49 (WeightLoader spec) + lines 82-91 (canonical Swift)
- Sortie 1a output files (ModuleParameters, KeyMapping, TensorTransform, QuantizationConfig)

**Dependencies**: Sortie 1a (needs `ModuleParameters`, `KeyMapping`, `TensorTransform`, `QuantizationConfig`).

**Exit Criteria**:
- [ ] `WeightLoader.load()` signature matches canonical definition: `static func load(componentId:keyMapping:tensorTransform:quantization:) async throws -> ModuleParameters`
- [ ] Loading pipeline: `withComponentAccess` -> find `.safetensors` URLs -> parse -> remap keys -> transform tensors -> quantize -> return `ModuleParameters`
- [ ] Key remapping: applies `KeyMapping` closure; keys returning `nil` are skipped
- [ ] Tensor transforms: applies `TensorTransform` closure when non-nil, after key remapping, before quantization
- [ ] Quantization: supports `.asStored`, `.float16`, `.bfloat16`, `.int4(groupSize:)`, `.int8(groupSize:)` via `QuantizationConfig`
- [ ] Handles sharded safetensors (multiple `.safetensors` files in component directory)
- [ ] Never exposes file paths or URLs outside `withComponentAccess` scope
- [ ] Throws `PipelineError.weightLoadingFailed` on parse or I/O errors `[OQ-1]`
- [ ] Throws `PipelineError.modelNotDownloaded` if component is not ready in Acervo
- [ ] Unit tests with synthetic safetensors data validate key remapping, tensor transform, and quantization

**Notes**:
- MLX Swift already has safetensors loading utilities (`MLX.loadArrays`). Evaluate whether to use those directly or wrap them. The key remapping and transform steps are SwiftTuberia-specific. `[OQ-1]`
- The WeightLoader is the ONLY code that reads safetensors. No pipe segment ever touches file I/O.
- Progressive loading (streaming keys) is a performance optimization that can be deferred to a follow-up if needed.

---

## Sortie 3: MemoryManager + DeviceCapability (Wave 1.3)

**Model**: sonnet
**Estimated tokens**: ~16K (two files with clear canonical definitions)
**Priority**: P0 (blocks Sorties 4b, 5b)

**Objective**: Implement device capability detection and the memory management actor that tracks loaded components and coordinates memory budgets across pipelines.

**Files**:
- `Sources/Tuberia/Infrastructure/DeviceCapability.swift` — `DeviceCapability` struct with `AppleSiliconGeneration`, `Platform` enums, `static let current`
- `Sources/Tuberia/Infrastructure/MemoryManager.swift` — `MemoryManager` actor with soft/hard checks, component tracking, GPU cache clearing

**Reference material for agent**:
- requirements/INFRASTRUCTURE.md lines 54-73 (MemoryManager + DeviceCapability spec) + lines 93-137 (canonical Swift)

**Dependencies**: Sortie 1a (protocol types exist for `PipelineError` reference — but note `PipelineError` is defined in Sortie 4a; MemoryManager can initially define its own error or depend on 4a).

**Exit Criteria**:
- [ ] `DeviceCapability.current` is synchronous, cached at first access, returns chip generation + total memory + platform + neural accelerator detection
- [ ] `AppleSiliconGeneration` enum covers M1 through M5, all tiers (base, Pro, Max, Ultra), plus `.unknown`
- [ ] `Platform` enum: `.macOS`, `.iPadOS`
- [ ] Chip detection uses `sysctlbyname("machdep.cpu.brand_string")`
- [ ] `MemoryManager.shared` is a global singleton actor
- [ ] `availableMemory` uses Mach VM statistics (free + inactive + purgeable + speculative pages)
- [ ] `softCheck(requiredBytes:)` returns `Bool` — checks if available memory exceeds requirement
- [ ] `hardValidate(requiredBytes:)` throws `PipelineError.insufficientMemory` if budget exceeded
- [ ] `registerLoaded(component:bytes:)` and `unregisterLoaded(component:)` track loaded component memory
- [ ] `loadedComponentsMemory` returns total tracked bytes
- [ ] `clearGPUCache()` calls MLX's GPU cache clear
- [ ] `deviceCapability` property on MemoryManager returns same value as `DeviceCapability.current`
- [ ] Unit tests validate device detection (at least on current machine), memory tracking arithmetic

**Notes**:
- MemoryManager does NOT auto-unload components. It reports state; the caller decides eviction policy.
- Headroom multipliers are per-consumer (applied externally), not in MemoryManager.
- Cross-pipeline tracking means image and TTS pipelines share the same MemoryManager instance.

---

## Sortie 4a: PipelineProgress + PipelineError (Wave 1.4a)

**Model**: haiku
**Estimated tokens**: ~3K (two small enums, copy-paste from canonical)
**Priority**: P0 (blocks 4b, 5a, 5b, 5c)

**Objective**: Implement the progress reporting enum and the full error model. These are simple enums copied from canonical definitions.

**Files**:
- `Sources/Tuberia/Pipeline/PipelineProgress.swift` — `PipelineProgress` enum
- `Sources/Tuberia/Pipeline/PipelineError.swift` — `PipelineError` enum

**Reference material for agent**:
- requirements/PIPELINE.md lines 420-442 (canonical Swift for PipelineProgress and PipelineError)

**Dependencies**: Sortie 1a (for `QuantizationConfig` import, though these enums are self-contained).

**Exit Criteria**:
- [ ] `PipelineProgress` enum matches canonical: `.downloading`, `.loading`, `.encoding`, `.generating`, `.decoding`, `.rendering`, `.complete`
- [ ] `PipelineError` enum matches canonical: all assembly, infrastructure, generation, and cancellation cases
- [ ] Both enums compile in the `Tuberia` target

---

## Sortie 4b: LoRA System (Wave 1.4b)

**Model**: sonnet
**Estimated tokens**: ~18K (LoRA loader + config, needs WeightLoader patterns and canonical defs)
**Priority**: P1 (blocks 5b, 5c for full LoRA support; pipeline can be built without LoRA initially)

**Objective**: Implement the LoRA loading/application system. LoRA loading reuses WeightLoader patterns; merge applies adapter A/B matrices into base weights.

**Files**:
- `Sources/Tuberia/Pipeline/LoRAConfig.swift` — `LoRAConfig` struct (from canonical definition)
- `Sources/Tuberia/Pipeline/LoRALoader.swift` — LoRA weight loading from safetensors, merge into base weights, scale, unload (restore base)

**Reference material for agent**:
- requirements/PIPELINE.md lines 186-206 (LoRA spec) + lines 336-359 (canonical LoRAConfig)
- WeightLoader.swift from Sortie 2 (for safetensors loading patterns)
- MemoryManager.swift from Sortie 3 (for memory tracking of LoRA weights)

**Dependencies**: Sortie 2 (WeightLoader for safetensors parsing), Sortie 3 (MemoryManager for memory tracking), Sortie 4a (PipelineError for error cases).

**Exit Criteria**:
- [ ] `LoRAConfig` struct matches canonical definition: `componentId`, `localPath`, `scale`, `activationKeyword` with precondition that at least one ID source is non-nil
- [ ] LoRA loader can load adapter weights from safetensors (via Acervo component or local path)
- [ ] LoRA application merges adapter A/B matrices into base weights using the segment's `keyMapping`
- [ ] LoRA scale (0.0-1.0) is applied during merge
- [ ] LoRA unload restores original base weights
- [ ] Single active LoRA constraint enforced (v1)
- [ ] Unit tests: LoRA merge math with synthetic weights, scale application, unload restoration

**Notes**:
- LoRA key mapping reuses the backbone's `WeightedSegment.keyMapping`. No separate LoRA key mapping is needed.
- LoRA files follow standard convention: `layer.lora_A` and `layer.lora_B` key suffixes.
- Multiple LoRA support (sequential load/unload) can be added later; v1 is single LoRA per generation.

---

## Sortie 5a: GenerationPipeline Protocol + PipelineRecipe + Request/Result Types (Wave 1.5a)

**Model**: sonnet
**Estimated tokens**: ~12K (protocol definitions + data types from canonical, no implementation logic)
**Priority**: P0 (blocks 5b, 5c)

**Objective**: Define the GenerationPipeline protocol, PipelineRecipe protocol, and all supporting data types needed by DiffusionPipeline. This is the type-level layer only — no implementation.

**Files**:
- `Sources/Tuberia/Pipeline/GenerationPipeline.swift` — `GenerationPipeline` protocol, `MemoryRequirement` struct
- `Sources/Tuberia/Pipeline/PipelineRecipe.swift` — `PipelineRecipe` protocol, `PipelineRole` enum, `UnconditionalEmbeddingStrategy` enum
- `Sources/Tuberia/Pipeline/DiffusionGenerationRequest.swift` — `DiffusionGenerationRequest` struct
- `Sources/Tuberia/Pipeline/DiffusionGenerationResult.swift` — `DiffusionGenerationResult` struct

**Reference material for agent**:
- requirements/PIPELINE.md lines 262-416 (canonical Swift for all pipeline types)
- Sortie 1b protocol files (for associated type references)
- Sortie 4a files (PipelineProgress, PipelineError)

**Dependencies**: Sortie 1b (protocols), Sortie 4a (PipelineProgress, PipelineError).

**Exit Criteria**:
- [ ] `GenerationPipeline` protocol matches canonical: `generate(request:progress:)`, `loadModels(progress:)`, `unloadModels()`, `memoryRequirement`, `isLoaded`
- [ ] `MemoryRequirement` struct with `peakMemoryBytes` and `phasedMemoryBytes`
- [ ] `PipelineRecipe` protocol matches canonical: associated types for all five segments, config properties, `supportsImageToImage`, `unconditionalEmbeddingStrategy`, `allComponentIds`, `quantizationFor(_:)`, `validate()`
- [ ] `PipelineRole` enum: `.encoder`, `.scheduler`, `.backbone`, `.decoder`, `.renderer`
- [ ] `UnconditionalEmbeddingStrategy` enum: `.emptyPrompt`, `.zeroVector(shape:)`, `.none`
- [ ] `DiffusionGenerationRequest` struct matches canonical (all fields including LoRA and img2img)
- [ ] `DiffusionGenerationResult` struct matches canonical
- [ ] All types compile in the `Tuberia` target

---

## Sortie 5b: DiffusionPipeline — Init, Validation, Load, Unload (Wave 1.5b)

**Model**: opus
**Estimated tokens**: ~40K (complex actor implementation with validation logic, two-phase loading, MemoryManager integration)
**Priority**: P0 (blocks 5c, 6a)

**Objective**: Implement DiffusionPipeline's construction, assembly-time validation, model loading orchestration, and unloading. This is the structural half of the pipeline.

**Files**:
- `Sources/Tuberia/Pipeline/DiffusionPipeline.swift` — `DiffusionPipeline<E,S,B,D,R>` actor: `init(recipe:)`, `loadModels(progress:)`, `unloadModels()`, `memoryRequirement`, `isLoaded`

**Reference material for agent**:
- requirements/PIPELINE.md lines 117-183 (recipe validation, two-phase loading, construction API)
- requirements/PIPELINE.md lines 394-416 (canonical DiffusionPipeline declaration)
- Sortie 5a output files (GenerationPipeline, PipelineRecipe, MemoryRequirement)
- requirements/INFRASTRUCTURE.md lines 82-110 (WeightLoader and MemoryManager canonical)

**Dependencies**: Sorties 1b, 2, 3, 4a, 4b (LoRAConfig), 5a.

**Exit Criteria**:
- [ ] `DiffusionPipeline<E,S,B,D,R>` is an actor conforming to `GenerationPipeline`
- [ ] `init(recipe:)` instantiates all five components via `init(configuration:)`, calls `recipe.validate()`, throws on failure
- [ ] Assembly-time validation: 6 checks (completeness, encoder-backbone dim, encoder-backbone seq, backbone-decoder channels, decoder-renderer modality, img2img BidirectionalDecoder) `[OQ-5]`
- [ ] `loadModels()` orchestrates: Acervo ensure ready -> WeightLoader.load() per weighted segment -> segment.apply(weights:) -> MemoryManager.registerLoaded()
- [ ] Two-phase loading: Phase 1 = TextEncoder (encode, then unload), Phase 2 = Backbone + Decoder + Renderer, with `clearGPUCache()` between phases
- [ ] `memoryRequirement` reports both `peakMemoryBytes` and `phasedMemoryBytes`
- [ ] `nonisolated` accessors for `memoryRequirement` and `isLoaded`
- [ ] `unloadModels()` unloads all weighted segments and unregisters from MemoryManager
- [ ] `generate()` method stubbed as `fatalError("Implemented in Sortie 5c")` to allow compilation

**Notes**:
- The `generate()` stub allows the pipeline to compile and be tested for init/validation/load/unload before the orchestration logic is added in 5c.
- `memoryRequirement` and `isLoaded` are `nonisolated` per the canonical definition.

---

## Sortie 5c: DiffusionPipeline — Generate Orchestration (Wave 1.5c)

**Model**: opus
**Estimated tokens**: ~35K (complex orchestration: encode -> latents -> denoising loop -> decode -> render, CFG, img2img, LoRA)
**Priority**: P0 (blocks 6a for full pipeline testing)

**Objective**: Implement DiffusionPipeline's `generate()` method — the full diffusion orchestration flow including CFG handling, img2img support, and LoRA lifecycle.

**Files**:
- `Sources/Tuberia/Pipeline/DiffusionPipeline.swift` — replace `generate()` stub with full implementation

**Reference material for agent**:
- requirements/PIPELINE.md lines 74-113 (CFG strategies, orchestration flow)
- requirements/PIPELINE.md lines 186-206 (LoRA application during generation)
- requirements/PROTOCOLS.md (TextEncoder/Scheduler/Backbone/Decoder/Renderer contracts)
- DiffusionPipeline.swift from Sortie 5b (existing init/load/unload code)

**Dependencies**: Sortie 5b (DiffusionPipeline actor structure).

**Exit Criteria**:
- [ ] `generate()` orchestration: encode -> initial latents -> scheduler configure -> denoising loop (CFG if applicable) -> decode -> render
- [ ] CFG handling per `UnconditionalEmbeddingStrategy`: `.emptyPrompt` encodes "", `.zeroVector` constructs zeros, `.none` skips CFG
- [ ] Image-to-image: if `referenceImages` present, encode via BidirectionalDecoder, addNoise, truncated schedule
- [ ] Field mapping: `TextEncoderOutput.embeddings` -> `BackboneInput.conditioning`, `.mask` -> `.conditioningMask`
- [ ] LoRA: if request includes `loRA`, load and apply before generation, unload after
- [ ] Progress callbacks fire at each stage
- [ ] Seed handling: use provided seed or generate random; store actual seed in result
- [ ] Returns `DiffusionGenerationResult` with output, seed, steps, guidanceScale, duration

**Notes**:
- The pipeline maps TextEncoderOutput fields to BackboneInput fields internally — neither encoder nor backbone knows the other's field names.
- Timestep normalization: schedulers produce scalar timesteps; the pipeline normalizes to scalar before calling `backbone.forward()`. Backbones must accept both scalar and `[B]` shapes defensively.

---

## Sortie 6a: Mock Components + Contract Tests (Wave 1.6a)

**Model**: opus
**Estimated tokens**: ~45K (7 mock files + 2 contract test files, needs full protocol knowledge)
**Priority**: P0 (validates entire Tuberia target)

**Objective**: Implement mock components for all pipe segment protocols and contract tests that validate pipe segment compatibility, assembly-time shape validation, and pipeline error behavior. All synthetic — no real weights or GPU.

**Files**:
- `Tests/TuberiaTests/Mocks/MockTextEncoder.swift` — configurable mock with settable `outputEmbeddingDim`, `maxSequenceLength`
- `Tests/TuberiaTests/Mocks/MockScheduler.swift` — configurable mock
- `Tests/TuberiaTests/Mocks/MockBackbone.swift` — configurable mock with settable `expectedConditioningDim`, `outputLatentChannels`, `expectedMaxSequenceLength`
- `Tests/TuberiaTests/Mocks/MockDecoder.swift` — configurable mock with settable `expectedInputChannels`; optional `BidirectionalDecoder` conformance
- `Tests/TuberiaTests/Mocks/MockRenderer.swift` — configurable mock
- `Tests/TuberiaTests/Mocks/MockPipelineRecipe.swift` — configurable mock recipe
- `Tests/TuberiaTests/ContractTests/ShapeValidationTests.swift` — assembly-time shape contract tests
- `Tests/TuberiaTests/ContractTests/PipelineAssemblyTests.swift` — assembly success and failure paths
- Remove `Tests/TuberiaTests/TuberiaTests.swift` (placeholder)

**Reference material for agent**:
- requirements/PROTOCOLS.md (all protocol definitions for mock conformance)
- requirements/PIPELINE.md lines 117-130 (6 assembly-time validation checks)
- requirements/TESTING.md (testing strategy and rules)
- DiffusionPipeline.swift from Sorties 5b+5c

**Dependencies**: Sorties 1b, 5a, 5b, 5c (all Tuberia pipeline code).

**Exit Criteria**:
- [ ] Mock components implement all protocol requirements with configurable shape values
- [ ] Mock conformers are `final class` with `@unchecked Sendable`
- [ ] Test: compatible encoder (dim=4096) + backbone (expectedDim=4096) assembles successfully
- [ ] Test: incompatible encoder (dim=4096) + backbone (expectedDim=768) throws `PipelineError.incompatibleComponents` with message mentioning "embedding dimension"
- [ ] Test: incompatible sequence length throws `incompatibleComponents`
- [ ] Test: incompatible backbone-decoder channels throws `incompatibleComponents`
- [ ] Test: recipe with `supportsImageToImage=true` and non-BidirectionalDecoder throws `incompatibleComponents`
- [ ] Test: recipe with `supportsImageToImage=true` and BidirectionalDecoder assembles successfully
- [ ] Test: `WeightedSegment.apply(weights:)` with correct keys sets `isLoaded = true`
- [ ] Test: `WeightedSegment.apply(weights:)` with missing keys throws clear error `[OQ-4]`
- [ ] Test: `WeightedSegment.unload()` sets `isLoaded = false` and clears weights
- [ ] All tests use synthetic inputs — no network, no real weights, no GPU
- [ ] No timed tests (no sleep, no wall-clock assertions)

---

## Sortie 6b: Infrastructure Unit Tests (Wave 1.6b)

**Model**: opus
**Estimated tokens**: ~30K (3 test files covering WeightLoader, MemoryManager, LoRA with synthetic data)
**Priority**: P0 (validates infrastructure before catalog integration)

**Objective**: Unit tests for WeightLoader, MemoryManager, and LoRA system using synthetic data.

**Files**:
- `Tests/TuberiaTests/WeightLoaderTests.swift` — key remapping, tensor transform, quantization tests with synthetic safetensors
- `Tests/TuberiaTests/MemoryManagerTests.swift` — tracking, soft/hard checks
- `Tests/TuberiaTests/LoRATests.swift` — merge math, scale, unload restoration

**Reference material for agent**:
- requirements/TESTING.md (testing rules)
- requirements/INFRASTRUCTURE.md (WeightLoader, MemoryManager specs)
- WeightLoader.swift, MemoryManager.swift, LoRALoader.swift from Sorties 2, 3, 4b

**Dependencies**: Sorties 2, 3, 4b (all infrastructure code).

**Exit Criteria**:
- [ ] Test: `WeightLoader.load()` applies keyMapping correctly (synthetic safetensors)
- [ ] Test: `WeightLoader.load()` applies tensorTransform correctly
- [ ] Test: `WeightLoader.load()` applies QuantizationConfig correctly
- [ ] Test: MemoryManager `registerLoaded`/`unregisterLoaded` tracks bytes accurately
- [ ] Test: MemoryManager `softCheck` returns false when budget exceeded
- [ ] Test: MemoryManager `hardValidate` throws `insufficientMemory` when budget exceeded
- [ ] Test: LoRA merge produces correct output (synthetic A/B matrices, known expected result)
- [ ] Test: LoRA scale=0.0 produces base weights unchanged
- [ ] Test: LoRA unload restores exact base weights
- [ ] All tests use synthetic inputs — no network, no real weights, no GPU
- [ ] No timed tests
- [ ] Coverage >= 90% for infrastructure code in `Tuberia` target

---

## Sortie 7: ImageRenderer (Wave 2.1)

**Model**: haiku
**Estimated tokens**: ~6K (simple stateless renderer, CGImage construction)
**Priority**: P1 (validates Renderer protocol end-to-end)

**Objective**: Implement the simplest catalog component — a stateless renderer that converts `DecodedOutput` (float pixel data as MLXArray) into `CGImage`.

**Files**:
- `Sources/TuberiaCatalog/Renderers/ImageRenderer.swift` — `ImageRenderer` conforming to `Renderer`, `Configuration = Void`
- `Tests/TuberiaCatalogTests/ImageRendererTests.swift` — synthetic pixel arrays -> valid CGImage with correct dimensions

**Reference material for agent**:
- requirements/PROTOCOLS.md lines 445-482 (Renderer protocol, RenderedOutput, AudioData, VideoFrames)
- requirements/CATALOG.md lines 42-48 (Renderers table)

**Dependencies**: Sortie 1b (Renderer protocol, DecodedOutput, RenderedOutput).

**Exit Criteria**:
- [ ] `ImageRenderer` conforms to `Renderer` with `Configuration = Void`
- [ ] `init(configuration: Void)` — trivial, no state
- [ ] `render(_:)` converts `DecodedOutput.data` ([B, H, W, 3] float in 0.0-1.0) to `CGImage`
- [ ] Handles float -> UInt8 conversion (clamp + scale by 255)
- [ ] Returns `.image(CGImage)` via `RenderedOutput`
- [ ] Test: known 2x2 pixel array produces CGImage with correct dimensions (2x2)
- [ ] Test: known pixel values are correctly represented in output CGImage
- [ ] Test: batch dimension (B > 1) produces first image `[OQ-6]`
- [ ] Remove `Tests/TuberiaCatalogTests/TuberiaCatalogTests.swift` (placeholder)

**Notes**:
- Renderers are stateless, weightless, freely concurrent. No `WeightedSegment` conformance.
- Uses `CoreGraphics` for CGImage construction (`CGContext`, `CGBitmapInfo`).

---

## Sortie 8: SDXLVAEDecoder (Wave 2.2)

**Model**: sonnet
**Estimated tokens**: ~28K (VAE architecture with ResNet blocks, attention, upsampling + configuration + tests)
**Priority**: P1 (first weighted catalog component, validates WeightedSegment lifecycle end-to-end)

**Objective**: Implement the SDXL VAE decoder that decodes 4-channel latents into pixel data.

**Files**:
- `Sources/TuberiaCatalog/Decoders/SDXLVAEDecoder.swift` — `SDXLVAEDecoder` conforming to `Decoder` (and optionally `BidirectionalDecoder`) `[OQ-2]`
- `Sources/TuberiaCatalog/Decoders/SDXLVAEDecoderConfiguration.swift` — `SDXLVAEDecoderConfiguration` struct (from canonical definitions)
- `Tests/TuberiaCatalogTests/SDXLVAEDecoderTests.swift` — shape validation, scaling factor application

**Reference material for agent**:
- requirements/CATALOG.md lines 94-105 (SDXLVAEDecoderConfiguration canonical)
- requirements/PROTOCOLS.md lines 384-443 (Decoder, BidirectionalDecoder protocols)
- WeightLoader.swift from Sortie 2 (for weight access pattern understanding)

**Dependencies**: Sortie 1b (Decoder protocol, WeightedSegment), Sortie 2 (WeightLoader for weight access pattern).

**Exit Criteria**:
- [ ] `SDXLVAEDecoder` conforms to `Decoder` with `Configuration = SDXLVAEDecoderConfiguration`
- [ ] `SDXLVAEDecoderConfiguration` matches canonical: `componentId`, `latentChannels` (default 4), `scalingFactor` (default 0.13025)
- [ ] `expectedInputChannels` returns `configuration.latentChannels`
- [ ] `scalingFactor` returns `configuration.scalingFactor`
- [ ] `decode(_:)` applies `latents * (1.0 / scalingFactor)` internally, then runs VAE decode forward pass
- [ ] `decode(_:)` returns `DecodedOutput` with `ImageDecoderMetadata`
- [ ] `keyMapping` provides safetensors-to-module key translation for SDXL VAE architecture
- [ ] `apply(weights:)` loads parameters into internal modules, sets `isLoaded = true`
- [ ] `unload()` clears parameters, sets `isLoaded = false`
- [ ] `estimatedMemoryBytes` returns approximate size (~160 MB for fp16)
- [ ] Implements actual VAE decoder architecture (ResNet blocks, attention, upsampling)
- [ ] Unit tests validate shape contracts with synthetic inputs
- [ ] Integration test (gated by `#if INTEGRATION_TESTS`) validates PSNR against reference

**Notes**:
- The VAE architecture implementation is the most substantial piece of this sortie. Reference existing flux-2-swift-mlx or pixart-swift-mlx VAE implementations for architecture guidance.
- `BidirectionalDecoder` conformance (adding `encode(_:)`) enables img2img. `[OQ-2]`

---

## Sortie 9: DPMSolverScheduler (Wave 2.3)

**Model**: sonnet
**Estimated tokens**: ~20K (mathematical algorithm + configuration + tests)
**Priority**: P1 (first scheduler, validates Scheduler protocol)

**Objective**: Implement the DPM-Solver++ multistep scheduler.

**Files**:
- `Sources/TuberiaCatalog/Schedulers/DPMSolverScheduler.swift` — `DPMSolverScheduler` conforming to `Scheduler`
- `Sources/TuberiaCatalog/Schedulers/DPMSolverSchedulerConfiguration.swift` — `DPMSolverSchedulerConfiguration` struct (from canonical definitions)
- `Tests/TuberiaCatalogTests/DPMSolverSchedulerTests.swift` — trajectory validation with synthetic noise predictions

**Reference material for agent**:
- requirements/CATALOG.md lines 69-81 (DPMSolverSchedulerConfiguration canonical)
- requirements/PROTOCOLS.md lines 272-332 (Scheduler protocol, BetaSchedule, PredictionType, SchedulerPlan)

**Dependencies**: Sortie 1b (Scheduler protocol, BetaSchedule, PredictionType, SchedulerPlan).

**Exit Criteria**:
- [ ] `DPMSolverScheduler` conforms to `Scheduler` with `Configuration = DPMSolverSchedulerConfiguration`
- [ ] `DPMSolverSchedulerConfiguration` matches canonical: `betaSchedule`, `predictionType`, `solverOrder`, `trainTimesteps`
- [ ] `configure(steps:startTimestep:)` computes correct timestep schedule based on beta schedule
- [ ] `configure(steps:startTimestep:)` with non-nil `startTimestep` truncates the plan (img2img support)
- [ ] `step(output:timestep:sample:)` implements DPM-Solver++ algorithm (first-order and second-order based on `solverOrder`)
- [ ] Supports `PredictionType.epsilon` and `PredictionType.velocity`
- [ ] `addNoise(to:noise:at:)` adds noise at the specified timestep level
- [ ] `reset()` clears internal state (previous step history for multistep)
- [ ] `BetaSchedule.linear` computes correct beta values
- [ ] `BetaSchedule.cosine` and `.sqrt` compute correct beta values
- [ ] No weights — pure mathematical computation
- [ ] Test: known noise schedule produces expected timesteps and sigmas
- [ ] Test: synthetic denoising trajectory matches expected values for DPM-Solver++ 2nd order
- [ ] Test: `startTimestep` truncation produces correct shortened schedule
- [ ] Test: `reset()` clears state, allowing clean re-run

**Notes**:
- DPM-Solver++ is well-documented in literature. Reference the diffusers Python implementation for algorithm details.
- Schedulers are NOT `WeightedSegment` — they have no weights. They ARE `Sendable`.
- The scheduler is stateful within a generation (tracks previous step outputs for multistep) but stateless between generations (via `reset()`).

---

## Sortie 10: T5XXLEncoder (Wave 2.4)

**Model**: sonnet
**Estimated tokens**: ~30K (T5 transformer architecture + tokenizer integration + configuration + tests)
**Priority**: P1 (largest catalog component, validates TextEncoder protocol end-to-end)

**Objective**: Implement the T5-XXL text encoder producing 4096-dim embeddings.

**Files**:
- `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` — `T5XXLEncoder` conforming to `TextEncoder`
- `Sources/TuberiaCatalog/Encoders/T5XXLEncoderConfiguration.swift` — `T5XXLEncoderConfiguration` struct (from canonical definitions)
- `Sources/TuberiaCatalog/Encoders/T5Model.swift` — T5 transformer encoder architecture (attention, FFN, layer norm)
- `Tests/TuberiaCatalogTests/T5XXLEncoderTests.swift` — shape validation, embedding dimension checks

**Reference material for agent**:
- requirements/CATALOG.md lines 56-67 (T5XXLEncoderConfiguration canonical)
- requirements/PROTOCOLS.md lines 227-270 (TextEncoder protocol, TextEncoderInput, TextEncoderOutput)
- requirements/CATALOG.md lines 8-17 (T5XXL component spec)

**Dependencies**: Sortie 1b (TextEncoder protocol, WeightedSegment), Sortie 2 (WeightLoader).

**Exit Criteria**:
- [ ] `T5XXLEncoder` conforms to `TextEncoder` with `Configuration = T5XXLEncoderConfiguration`
- [ ] `T5XXLEncoderConfiguration` matches canonical: `componentId` (default "t5-xxl-encoder-int4"), `maxSequenceLength` (default 120), `embeddingDim` (4096)
- [ ] `init(configuration:)` loads tokenizer from Acervo component via `withComponentAccess` using `AutoTokenizer.from(modelFolder:)`
- [ ] `outputEmbeddingDim` returns `configuration.embeddingDim` (4096)
- [ ] `maxSequenceLength` returns `configuration.maxSequenceLength`
- [ ] `encode(_:)` tokenizes input text, runs T5 encoder forward pass, returns `TextEncoderOutput` with `[B, seq, 4096]` embeddings and `[B, seq]` mask
- [ ] Mask: 1 = real token, 0 = padding
- [ ] `keyMapping` provides safetensors-to-module key translation for T5-XXL architecture
- [ ] `apply(weights:)` loads parameters, sets `isLoaded = true`
- [ ] `unload()` clears parameters, sets `isLoaded = false`
- [ ] `estimatedMemoryBytes` returns approximate size (~1.2 GB for int4)
- [ ] Implements T5 encoder architecture (self-attention, relative position bias, feed-forward, RMS layer norm)
- [ ] Unit tests: mock tokenizer input -> correct output shape [1, seq, 4096]
- [ ] Integration test (gated): real weights -> known prompt -> expected embedding shape and approximate values

**Notes**:
- T5-XXL is large (~1.2 GB int4). The tokenizer is bundled with weights in the same Acervo component.
- The T5 architecture uses relative position biases, RMS layer norm, and gated FFN.
- The `swift-transformers` dependency is needed here for `AutoTokenizer`.

---

## Sortie 11: FlowMatchEulerScheduler (Wave 2.5)

**Model**: sonnet
**Estimated tokens**: ~14K (mathematical algorithm, simpler than DPM-Solver)
**Priority**: P2 (validates scheduler protocol flexibility, needed for future FLUX support)

**Objective**: Implement the flow matching Euler scheduler for FLUX-family models.

**Files**:
- `Sources/TuberiaCatalog/Schedulers/FlowMatchEulerScheduler.swift` — `FlowMatchEulerScheduler` conforming to `Scheduler`
- `Sources/TuberiaCatalog/Schedulers/FlowMatchEulerSchedulerConfiguration.swift` — `FlowMatchEulerSchedulerConfiguration` struct (from canonical definitions)
- `Tests/TuberiaCatalogTests/FlowMatchEulerSchedulerTests.swift` — sigma schedule validation, step trajectory

**Reference material for agent**:
- requirements/CATALOG.md lines 83-91 (FlowMatchEulerSchedulerConfiguration canonical)
- requirements/PROTOCOLS.md lines 272-332 (Scheduler protocol)

**Dependencies**: Sortie 1b (Scheduler protocol).

**Exit Criteria**:
- [ ] `FlowMatchEulerScheduler` conforms to `Scheduler` with `Configuration = FlowMatchEulerSchedulerConfiguration`
- [ ] `FlowMatchEulerSchedulerConfiguration` matches canonical: `shift` (default 1.0)
- [ ] `configure(steps:startTimestep:)` computes sigma schedule via rectified flow formulation
- [ ] `step(output:timestep:sample:)` implements Euler step for flow matching
- [ ] `addNoise(to:noise:at:)` adds noise at the specified sigma level
- [ ] `reset()` clears state
- [ ] Does NOT use `BetaSchedule` — flow matching uses sigma schedules directly
- [ ] No weights — pure mathematical computation
- [ ] Test: known step count produces expected sigma schedule
- [ ] Test: Euler step with known inputs produces expected output
- [ ] Test: `shift` parameter correctly adjusts the sigma schedule

---

## Sortie 12: AudioRenderer (Wave 2.6)

**Model**: haiku
**Estimated tokens**: ~6K (simple stateless renderer, WAV format construction)
**Priority**: P2 (future audio diffusion models)

**Objective**: Implement the audio renderer that converts decoded audio samples into WAV data.

**Files**:
- `Sources/TuberiaCatalog/Renderers/AudioRenderer.swift` — `AudioRenderer` conforming to `Renderer`, `Configuration = Void`
- `Tests/TuberiaCatalogTests/AudioRendererTests.swift` — synthetic sample array -> valid WAV data with correct format

**Reference material for agent**:
- requirements/PROTOCOLS.md lines 445-482 (Renderer protocol, AudioData)
- requirements/CATALOG.md lines 42-48 (Renderers table)

**Dependencies**: Sortie 1b (Renderer protocol, AudioData, RenderedOutput).

**Exit Criteria**:
- [ ] `AudioRenderer` conforms to `Renderer` with `Configuration = Void`
- [ ] `render(_:)` converts `DecodedOutput.data` ([B, samples] float) to WAV `Data`
- [ ] Extracts `sampleRate` from `AudioDecoderMetadata` in `DecodedOutput.metadata`
- [ ] Returns `.audio(AudioData)` via `RenderedOutput`
- [ ] WAV header is correct (RIFF format, PCM, correct sample rate and bit depth)
- [ ] Handles float -> int16 sample conversion (clamp + scale)
- [ ] Test: known sample array produces WAV data with correct header
- [ ] Test: correct sample rate from metadata is reflected in WAV header
- [ ] Test: output Data is playable (valid WAV format)

**Notes**:
- Like ImageRenderer, AudioRenderer is stateless and weightless. Configuration is `Void`.
- WAV format is straightforward: 44-byte RIFF header + PCM sample data.

---

## Sortie 13: Acervo Self-Registration + Catalog Integration (Wave 2.7)

**Model**: sonnet
**Estimated tokens**: ~12K (registration wiring + re-exports + tests)
**Priority**: P2 (final integration wiring)

**Objective**: Wire up TuberiaCatalog's self-registration of Acervo `ComponentDescriptor` entries at import time, and ensure the catalog module re-exports necessary Tuberia types cleanly.

**Files**:
- `Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` — static `let` initialization that registers T5-XXL and SDXL VAE descriptors with Acervo
- `Sources/TuberiaCatalog/TuberiaCatalog.swift` — update to `@_exported import Tuberia` if appropriate, or explicit re-exports
- `Tests/TuberiaCatalogTests/CatalogRegistrationTests.swift` — verify descriptors are registered after import

**Reference material for agent**:
- requirements/INFRASTRUCTURE.md lines 8-19 (Acervo integration spec)
- requirements/CATALOG.md (component specs with component IDs)

**Dependencies**: Sorties 7-12 (all catalog components exist).

**Exit Criteria**:
- [ ] Importing `TuberiaCatalog` auto-registers Acervo descriptors for `t5-xxl-encoder-int4` and `sdxl-vae-decoder-fp16`
- [ ] Registration is idempotent — duplicate registration is silently ignored (Acervo dedup)
- [ ] Descriptors include correct HuggingFace repo, file patterns, estimated sizes
- [ ] SHA-256 checksums may be `nil` initially (backfilled after weight conversion) `[OQ-3]`
- [ ] Test: after importing TuberiaCatalog, `Acervo.registeredComponents()` includes the expected IDs
- [ ] Test: model plugin registering the same component ID does not cause an error
- [ ] Remove placeholder `TuberiaCatalog.swift` content (replace with registration or re-export)

**Notes**:
- Scheduler and renderer components have no Acervo descriptors (no weights to download).
- The `intrusive-memory/t5-xxl-int4-mlx` and `intrusive-memory/sdxl-vae-fp16-mlx` HuggingFace repos must exist before integration tests can pass. Unit tests should mock Acervo. `[OQ-3]`

---

## Dependency Graph

```
Sortie 0  (recon) [COMPLETE]
    |
Sortie 1a (Package.swift + data types)  [haiku]
    |
Sortie 1b (protocols)  [sonnet]
    |
    +---> Sortie 2  (WeightLoader)        [sonnet]   ─┐
    |                                                   |
    +---> Sortie 3  (MemoryManager)       [sonnet]   ─┤
    |                                                   |
    +---> Sortie 4a (Progress + Errors)   [haiku]    ─┤
    |         |                                         |
    |         +---> Sortie 5a (Pipeline types)  [sonnet]
    |                   |                               |
    +---------+---> Sortie 4b (LoRA)      [sonnet]   ─┤
                        |                               |
                    Sortie 5b (Pipeline init/load) [opus]
                        |
                    Sortie 5c (Pipeline generate)  [opus]
                        |
                    Sortie 6a (Mocks + Contract tests) [opus]

Sortie 2 + 3 + 4a ---> Sortie 6b (Infrastructure tests) [opus]

Sortie 1b ---> Sortie 7  (ImageRenderer)        [haiku]   ─┐
Sortie 1b ---> Sortie 9  (DPMSolverScheduler)   [sonnet]  ─┤ parallelizable
Sortie 1b ---> Sortie 11 (FlowMatchEuler)       [sonnet]  ─┤ with 2-6
Sortie 1b ---> Sortie 12 (AudioRenderer)        [haiku]   ─┘

Sortie 1b + 2 ---> Sortie 8  (SDXLVAEDecoder)   [sonnet]
Sortie 1b + 2 ---> Sortie 10 (T5XXLEncoder)     [sonnet]

Sorties 7-12 ---> Sortie 13 (Catalog Registration) [sonnet]
```

---

## Parallelism Map (max 4 sub-agents)

### Phase A: Foundation (sequential, single agent)
```
1a [haiku] -> 1b [sonnet]
```

### Phase B: Infrastructure + Catalog Fork (up to 4 agents)

After Sortie 1b completes, launch up to 4 parallel tracks:

| Agent | Track | Sorties | Model |
|---|---|---|---|
| Agent 1 | Infrastructure-Core | 2 (WeightLoader) -> 4b (LoRA) | sonnet -> sonnet |
| Agent 2 | Infrastructure-Memory | 3 (MemoryManager) | sonnet |
| Agent 3 | Types + Lightweight Catalog | 4a (enums) [haiku] -> 7 (ImageRenderer) [haiku] -> 12 (AudioRenderer) [haiku] | haiku |
| Agent 4 | Catalog-Schedulers | 9 (DPMSolver) [sonnet] -> 11 (FlowMatch) [sonnet] | sonnet |

### Phase C: Pipeline Assembly (sequential on Agent 1, others continue catalog)

After Agents 1+2+3 complete Phase B:

| Agent | Track | Sorties | Model |
|---|---|---|---|
| Agent 1 | Pipeline | 5a [sonnet] -> 5b [opus] -> 5c [opus] | sonnet -> opus -> opus |
| Agent 2 | Infrastructure Tests | 6b [opus] | opus |
| Agent 3 | Weighted Catalog | 8 (SDXL VAE) [sonnet] (after Agent 1 finishes Sortie 2) | sonnet |
| Agent 4 | Weighted Catalog | 10 (T5XXL) [sonnet] (after Agent 1 finishes Sortie 2) | sonnet |

### Phase D: Validation + Registration (after all above)

| Agent | Track | Sorties | Model |
|---|---|---|---|
| Agent 1 | Contract Tests | 6a [opus] | opus |
| Agent 2 | Registration | 13 [sonnet] (after 7-12 complete) | sonnet |

---

## Priority Scoring

Each sortie scored on: Impact (1-5) x Urgency (1-5) x Unblock-Count (number of downstream sorties unblocked).

| Sortie | Impact | Urgency | Unblocks | Score | Priority |
|---|---|---|---|---|---|
| 1a | 5 | 5 | 15 | 375 | P0 |
| 1b | 5 | 5 | 13 | 325 | P0 |
| 2 | 5 | 5 | 6 | 150 | P0 |
| 3 | 4 | 5 | 4 | 80 | P0 |
| 4a | 3 | 5 | 5 | 75 | P0 |
| 4b | 4 | 4 | 3 | 48 | P1 |
| 5a | 5 | 4 | 3 | 60 | P0 |
| 5b | 5 | 4 | 2 | 40 | P0 |
| 5c | 5 | 4 | 1 | 20 | P0 |
| 6a | 5 | 3 | 0 | 0* | P0 |
| 6b | 4 | 3 | 0 | 0* | P0 |
| 7 | 3 | 3 | 1 | 9 | P1 |
| 8 | 4 | 3 | 1 | 12 | P1 |
| 9 | 3 | 3 | 1 | 9 | P1 |
| 10 | 4 | 3 | 1 | 12 | P1 |
| 11 | 2 | 2 | 1 | 4 | P2 |
| 12 | 2 | 2 | 1 | 4 | P2 |
| 13 | 3 | 2 | 0 | 0* | P2 |

*Scores of 0 for terminal sorties reflect no downstream unblocking, but 6a/6b are P0 because they validate the entire system.

---

## Open Questions

**[OQ-1] MLX safetensors utilities vs. custom parsing**: MLX Swift likely provides `MLX.loadArrays(url:)` or similar. Should WeightLoader wrap this (adding key remapping + transforms on top) or reimplement safetensors parsing? **Recommendation**: Wrap MLX's loader. The key remapping and transform steps are SwiftTuberia-specific additions, but the binary parsing should reuse MLX's battle-tested implementation. **Action required**: Agent implementing Sortie 2 must investigate `MLX.loadArrays` API surface before proceeding.

**[OQ-2] BidirectionalDecoder on SDXLVAEDecoder**: Should Sortie 8 include `BidirectionalDecoder` conformance (encode path) immediately, or defer to a follow-up? The encode path requires implementing the VAE encoder half, which roughly doubles the architecture code. **Recommendation**: Defer to a follow-up sortie (8b). Sortie 8 implements Decoder-only. This keeps Sortie 8 within sonnet's context budget. Img2img is not needed for the initial PixArt integration.

**[OQ-3] HuggingFace repo creation**: The Acervo component IDs reference HuggingFace repos (`intrusive-memory/t5-xxl-int4-mlx`, `intrusive-memory/sdxl-vae-fp16-mlx`) that need to be created and populated with converted weights before integration tests pass. **Action required**: This is an external dependency outside the execution plan. Unit tests must mock Acervo. Integration tests are gated by `#if INTEGRATION_TESTS`.

**[OQ-4] WeightedSegment missing keys behavior**: The exit criteria for 6a states "apply(weights:) with missing keys throws clear error" but the canonical `WeightedSegment` protocol defines `apply(weights:) throws` without specifying what constitutes "missing keys." **Recommendation**: Each concrete `WeightedSegment` implementation defines its required key set. `apply(weights:)` should throw `PipelineError.weightLoadingFailed(component:reason:)` listing the missing keys. This is an implementation concern per conformer, not a protocol concern.

**[OQ-5] Decoder-Renderer modality validation (check #5)**: The assembly validation requires checking that "decoder output modality is compatible with renderer input" but there is no modality type on the Decoder or Renderer protocols. **Recommendation**: Add a `Modality` enum (`.image`, `.audio`, `.video`) to both `DecoderMetadata` and `Renderer` protocol, or validate via runtime type checks (e.g., `metadata is ImageDecoderMetadata && renderer is ImageRenderer`). **Action required**: Resolve before implementing Sortie 5b.

**[OQ-6] Batch dimension handling in ImageRenderer**: The exit criteria say "batch dimension (B > 1) produces first image" but this is not specified in the requirements. **Recommendation**: Render only the first image in the batch (index 0). Document this explicitly. Multi-image batch rendering can be a future enhancement.

**[OQ-7] CLIPEncoder and DDPMScheduler**: Listed in CATALOG.md tables but not in the primary implementation order. **Decision**: Defer to a future wave unless needed by an imminent model plugin. No action needed for this execution plan.
