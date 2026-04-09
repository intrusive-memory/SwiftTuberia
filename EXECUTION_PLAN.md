---
feature_name: OPERATION BLUNT BLADE REFORGE
starting_point_commit: c3840ebf54379f10ff21554938ea67eb7b470dee
mission_branch: mission/blunt-blade-reforge/01
iteration: 1
---

# EXECUTION_PLAN.md — SwiftTuberia Test Coverage Gap Remediation

**Source**: `requirements/TEST_COVERAGE_GAPS.md`
**Date**: 2026-04-08
**Status**: READY FOR EXECUTION — all blocking issues resolved

---

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

---

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|-------------|
| TuberiaCatalogTests | Tests/TuberiaCatalogTests/ | 4 (1–4) | 1 | none |
| TuberiaTests | Tests/TuberiaTests/ | 1 (5) | 1 | none |
| TuberiaGPUTests | Tests/TuberiaGPUTests/ | 2 (6–7) | 1 | none |

All 3 work units are Layer 1 — fully parallel, no cross-unit dependencies.

---

## Work Unit: TuberiaCatalogTests

### Sortie 1: Remove 4 redundant tests and strengthen T5 apply-weights test

**Priority**: 12.5 — Foundation sortie; 3 downstream sorties (2, 3, 4) depend on TuberiaCatalogTests compiling clean. Highest priority in mission.

**Entry criteria**:
- [ ] First sortie — no prerequisites
- [ ] `Tests/TuberiaCatalogTests/T5KeyMappingTests.swift` contains `applyWeightsSetsIsLoaded` and `unloadResetsIsLoaded` (verify: `grep -c 'applyWeightsSetsIsLoaded\|unloadResetsIsLoaded' Tests/TuberiaCatalogTests/T5KeyMappingTests.swift` returns 2)
- [ ] `Tests/TuberiaCatalogTests/SDXLVAEModelTests.swift` contains `outputHas3Channels` and `residualConnectionShape` (verify: `grep -c 'outputHas3Channels\|residualConnectionShape' Tests/TuberiaCatalogTests/SDXLVAEModelTests.swift` returns 2)
- [ ] `Tests/TuberiaCatalogTests/T5KeyMappingTests.swift` contains `applyWeightsDoesNotCrash` (verify: `grep -c 'applyWeightsDoesNotCrash' Tests/TuberiaCatalogTests/T5KeyMappingTests.swift` returns 1)

**Tasks**:
1. Read `Tests/TuberiaCatalogTests/T5KeyMappingTests.swift` in full; delete the `applyWeightsSetsIsLoaded` test method from `T5ApplyWeightsTests` (REQ-TCG-001)
2. Delete the `unloadResetsIsLoaded` test method from `T5ApplyWeightsTests` in the same file (REQ-TCG-002); confirm `applyAndUnloadLifecycle` is retained
3. Read `Tests/TuberiaCatalogTests/SDXLVAEModelTests.swift` in full; delete `SDXLVAEDecoderModelTests.outputHas3Channels` (REQ-TCG-003); confirm `forwardPassShape8x8` is retained
4. Delete `AttentionBlockTests.residualConnectionShape` from `SDXLVAEModelTests.swift` (REQ-TCG-004); confirm `outputShapeMatchesInput` is retained
5. Return to `T5KeyMappingTests.swift`; locate `applyWeightsDoesNotCrash` within `T5ApplyWeightsTests`; after the existing `apply(weights:)` call, add an assertion that samples at least one known parameter (e.g. `embedding.weight`) and verifies it differs from zero-initialized values — use `@testable import TuberiaCatalog` to access internal state (REQ-TCG-060)
6. Confirm `applyWeightsDoesNotCrash` now contains at least 2 `#expect` calls (the crash-guard plus the new assertion)

**Exit criteria**:
- [ ] `grep 'applyWeightsSetsIsLoaded\|unloadResetsIsLoaded' Tests/TuberiaCatalogTests/T5KeyMappingTests.swift` returns no output
- [ ] `grep 'outputHas3Channels\|residualConnectionShape' Tests/TuberiaCatalogTests/SDXLVAEModelTests.swift` returns no output
- [ ] `grep 'applyAndUnloadLifecycle' Tests/TuberiaCatalogTests/T5KeyMappingTests.swift` returns a match (superset test retained)
- [ ] `grep 'forwardPassShape8x8' Tests/TuberiaCatalogTests/SDXLVAEModelTests.swift` returns a match
- [ ] `grep 'outputShapeMatchesInput' Tests/TuberiaCatalogTests/SDXLVAEModelTests.swift` returns a match
- [ ] `applyWeightsDoesNotCrash` now has at least 2 `#expect` calls (verify: `grep -A 25 'applyWeightsDoesNotCrash' Tests/TuberiaCatalogTests/T5KeyMappingTests.swift | grep -c '#expect'` returns ≥ 2)
- [ ] TuberiaCatalogTests compiles and passes: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaCatalogTests`

---

### Sortie 2: Add DPMSolver and FlowMatchEuler scheduler CPU unit tests

**Priority**: 3 — Independent after Sortie 1; higher risk than Sortie 3/4 due to MLX array math and scheduler API surface.

**Entry criteria**:
- [ ] Sortie 1 exit criteria met — TuberiaCatalogTests compiles and passes
- [ ] `Sources/TuberiaCatalog/Schedulers/DPMSolverScheduler.swift` and `FlowMatchEulerScheduler.swift` exist
- [ ] GPU reference files exist: `Tests/TuberiaCatalogGPUTests/DPMSolverSchedulerTests.swift`, `Tests/TuberiaCatalogGPUTests/FlowMatchEulerSchedulerTests.swift` (use for API style only — do NOT copy GPU tests into the CPU target)

**Tasks**:
1. Read `Sources/TuberiaCatalog/Schedulers/DPMSolverScheduler.swift` and `Sources/TuberiaCatalog/Schedulers/DPMSolverSchedulerConfiguration.swift` to understand `configure(numSteps:)`, plan structure (`timesteps`, `sigmas`), `step()`, and `addNoise(latents:noise:timestep:)` signatures
2. Read `Sources/TuberiaCatalog/Schedulers/FlowMatchEulerScheduler.swift` and `FlowMatchEulerSchedulerConfiguration.swift` to understand the same API surface plus sigma range contracts (first ≈ 1.0, last ≈ 0.0) and `stepWithZeroVelocityIsIdentity` contract
3. Read the MARK comment block header in `Tests/TuberiaCatalogTests/SDXLVAEModelTests.swift` (lines 1–27) for style reference on what to document in the new file headers
4. Create `Tests/TuberiaCatalogTests/DPMSolverSchedulerTests.swift` with:
   - MARK block at top documenting the gap this regresses (no GPU scheduler unit tests existed)
   - `@testable import TuberiaCatalog`; `@preconcurrency import MLX`
   - Suite `DPMSolverSchedulerConfigurationTests` (.serialized): 5 tests: `configureProducesCorrectStepCount`, `linearBetaScheduleIsDifferentFromCosine`, `sigmasAreMonotonicallyDecreasing`, `firstSigmaIsLargest`, `lastSigmaApproachesZero`
   - Suite `DPMSolverStepTests` (.serialized): 4 tests: `stepReducesNoiseMagnitude`, `stepPreservesShape`, `addNoiseIncreasesL2Norm`, `addNoisePreservesShape`
   - All inputs: synthetic `MLXArray` with known shapes; no real weights; no network; `eval()` before every assertion on array values
5. Create `Tests/TuberiaCatalogTests/FlowMatchEulerSchedulerTests.swift` with:
   - MARK block at top
   - Suite `FlowMatchEulerConfigurationTests` (.serialized): 4 tests: `configureProducesCorrectStepCount`, `sigmasSpanFullRange`, `sigmasAreMonotonicallyDecreasing`, `differentStepCountsProduceDifferentSchedules`
   - Suite `FlowMatchEulerStepTests` (.serialized): 3 tests: `stepPreservesShape`, `stepWithZeroVelocityIsIdentity`, `addNoisePreservesShape`
   - All inputs synthetic; `eval()` before assertions

**Exit criteria**:
- [ ] `Tests/TuberiaCatalogTests/DPMSolverSchedulerTests.swift` exists (verify: `test -f Tests/TuberiaCatalogTests/DPMSolverSchedulerTests.swift`)
- [ ] `Tests/TuberiaCatalogTests/FlowMatchEulerSchedulerTests.swift` exists (verify: `test -f Tests/TuberiaCatalogTests/FlowMatchEulerSchedulerTests.swift`)
- [ ] Each file contains exactly 2 suites (verify: `grep -c '@Suite' Tests/TuberiaCatalogTests/DPMSolverSchedulerTests.swift` returns 2; same for FlowMatchEuler file)
- [ ] Neither file uses `sleep`, `Thread.sleep`, or `Task.sleep` (verify: `grep -rn 'sleep' Tests/TuberiaCatalogTests/DPMSolverSchedulerTests.swift Tests/TuberiaCatalogTests/FlowMatchEulerSchedulerTests.swift` returns no output)
- [ ] Both files follow `eval()` pattern before assertions (verify: `grep -c 'eval()' Tests/TuberiaCatalogTests/DPMSolverSchedulerTests.swift` returns ≥ 4; `grep -c 'eval()' Tests/TuberiaCatalogTests/FlowMatchEulerSchedulerTests.swift` returns ≥ 3)
- [ ] Both files contain a MARK comment block at top (verify: `grep '// MARK:' Tests/TuberiaCatalogTests/DPMSolverSchedulerTests.swift` returns a match; `grep '// MARK:' Tests/TuberiaCatalogTests/FlowMatchEulerSchedulerTests.swift` returns a match)
- [ ] All 16 new tests compile and pass: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaCatalogTests`

---

### Sortie 3: Add SDXLVAEDecoder tensor-transform unit tests

**Priority**: 2.75 — Independent after Sortie 1; higher risk than Sortie 4 due to MLX tensor math and transposition verification.

**Entry criteria**:
- [ ] Sortie 1 exit criteria met — TuberiaCatalogTests compiles and passes
- [ ] `Tests/TuberiaCatalogTests/SDXLVAEDecoderTests.swift` exists (verify: `test -f Tests/TuberiaCatalogTests/SDXLVAEDecoderTests.swift`)
- [ ] `Sources/TuberiaCatalog/Decoders/SDXLVAEDecoder.swift` exists (verify: `test -f Sources/TuberiaCatalog/Decoders/SDXLVAEDecoder.swift`)

**Tasks**:
1. Read `Tests/TuberiaCatalogTests/SDXLVAEDecoderTests.swift` in full to understand existing suite structure and `SDXLVAEDecoder` API usage
2. Read `Sources/TuberiaCatalog/Decoders/SDXLVAEDecoder.swift` to understand the `tensorTransform` signature: takes a key `String` and value `MLXArray`, returns a transformed `MLXArray`
3. Add `SDXLVAEDecoderTensorTransformTests` suite to `Tests/TuberiaCatalogTests/SDXLVAEDecoderTests.swift` with 4 tests: `convWeightIsTransposed`, `biasPassesThroughUnchanged`, `nonConvWeightPassesThroughUnchanged`, `transpositionIsCorrect`; all use synthetic tensors; `eval()` before shape/value assertions

**Exit criteria**:
- [ ] `Tests/TuberiaCatalogTests/SDXLVAEDecoderTests.swift` contains `SDXLVAEDecoderTensorTransformTests` suite (verify: `grep 'SDXLVAEDecoderTensorTransformTests' Tests/TuberiaCatalogTests/SDXLVAEDecoderTests.swift`)
- [ ] `grep -c 'convWeightIsTransposed\|biasPassesThroughUnchanged\|nonConvWeightPassesThroughUnchanged\|transpositionIsCorrect' Tests/TuberiaCatalogTests/SDXLVAEDecoderTests.swift` returns 4
- [ ] No `sleep` usage added (verify: `grep 'sleep' Tests/TuberiaCatalogTests/SDXLVAEDecoderTests.swift` — new tests only)
- [ ] New suite follows `eval()` pattern before assertions (verify: `grep -A 60 'SDXLVAEDecoderTensorTransformTests' Tests/TuberiaCatalogTests/SDXLVAEDecoderTests.swift | grep -c 'eval()'` returns ≥ 2)
- [ ] All new tests compile and pass: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaCatalogTests`

---

### Sortie 4: Add ImageRenderer CPU unit tests

**Priority**: 1.75 — Independent after Sortie 1; lower risk (CoreGraphics, no MLX math).

**Entry criteria**:
- [ ] Sortie 1 exit criteria met — TuberiaCatalogTests compiles and passes
- [ ] `Sources/TuberiaCatalog/Renderers/ImageRenderer.swift` exists (verify: `test -f Sources/TuberiaCatalog/Renderers/ImageRenderer.swift`)

**Tasks**:
1. Read `Sources/TuberiaCatalog/Renderers/ImageRenderer.swift` to understand the public API: input shape `[1, H, W, 3]`, output `CGImage`, and pixel-range contracts
2. Read the MARK comment block header in `Tests/TuberiaCatalogTests/SDXLVAEModelTests.swift` (lines 1–27) for style reference
3. Create `Tests/TuberiaCatalogTests/ImageRendererUnitTests.swift` with:
   - MARK block at top documenting the float→UInt8 pixel conversion gap
   - `import CoreGraphics`; `@testable import TuberiaCatalog`
   - Suite `ImageRendererConversionTests`: 6 tests: `floatZeroMapsToUInt8Zero`, `floatOneMapsToUInt8_255`, `outOfRangeValuesAreClamped`, `outputImageHasCorrectDimensions`, `outputBitDepthIs8`, `outputColorSpaceisSRGB`
   - Tests use `CGImage` pixel sampling to verify values; no GPU required

**Exit criteria**:
- [ ] `Tests/TuberiaCatalogTests/ImageRendererUnitTests.swift` exists (verify: `test -f Tests/TuberiaCatalogTests/ImageRendererUnitTests.swift`)
- [ ] `grep 'ImageRendererConversionTests' Tests/TuberiaCatalogTests/ImageRendererUnitTests.swift` returns a match
- [ ] `grep -c '@Suite' Tests/TuberiaCatalogTests/ImageRendererUnitTests.swift` returns 1
- [ ] `grep 'sleep' Tests/TuberiaCatalogTests/ImageRendererUnitTests.swift` returns no output
- [ ] `grep '// MARK:' Tests/TuberiaCatalogTests/ImageRendererUnitTests.swift` returns a match
- [ ] All 6 new tests compile and pass: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaCatalogTests`

---

## Work Unit: TuberiaTests

### Sortie 5: Remove 2 redundant LoRA tests and add unknown-chip device capability tests

**Priority**: 1.75 — Standalone; no other sorties depend on it.

**Entry criteria**:
- [ ] First sortie — no prerequisites
- [ ] `Tests/TuberiaTests/LoRAIntegrationTests.swift` contains `testLoRAScaleZeroNoChange` and `testLoRAFractionalScale` (verify: `grep -c 'testLoRAScaleZeroNoChange\|testLoRAFractionalScale' Tests/TuberiaTests/LoRAIntegrationTests.swift` returns 2)
- [ ] `Tests/TuberiaTests/DeviceCapabilityTests.swift` exists (verify: `test -f Tests/TuberiaTests/DeviceCapabilityTests.swift`)

**Tasks**:
1. Read `Tests/TuberiaTests/LoRAIntegrationTests.swift` in full; locate `testLoRAScaleZeroNoChange`, `testLoRAFractionalScale`, and `testLoRARoundTripFractionalScale`; confirm `testLoRARoundTripFractionalScale` is present before making any changes (it must NOT be deleted — REQ-TCG-006 explicitly retains it)
2. Delete `testLoRAScaleZeroNoChange` from `LoRAIntegrationTests.swift` (REQ-TCG-005)
3. Delete `testLoRAFractionalScale` from `LoRAIntegrationTests.swift` (REQ-TCG-006); verify `testLoRARoundTripFractionalScale` still exists after the deletion
4. Read `Tests/TuberiaTests/DeviceCapabilityTests.swift` and `Sources/Tuberia/Infrastructure/DeviceCapability.swift` to understand `AppleSiliconGeneration` enum cases (including `.unknown`), the `parseChipGeneration(from:)` internal function, and the `hasNeuralAccelerators` property (note: the property is named `hasNeuralAccelerators`, NOT `hasNeuralEngine`)
5. Add `unknownChipStringReturnsUnknownGeneration` test to `DeviceCapabilityTests.swift`: pass `"apple m99"` through `DeviceCapability.parseChipGeneration(from:)`; assert it returns `.unknown` generation (REQ-TCG-061)
6. Add `unknownGenerationHasNoNeuralAccelerator` test to `DeviceCapabilityTests.swift`: construct a `DeviceCapability` with `chipGeneration: .unknown` using the memberwise initializer; assert `hasNeuralAccelerators == false` (REQ-TCG-061). Note: `detectNeuralAccelerators` is private; use the memberwise init to construct a capability that documents the contract.

**Exit criteria**:
- [ ] `grep 'testLoRAScaleZeroNoChange\|testLoRAFractionalScale' Tests/TuberiaTests/LoRAIntegrationTests.swift` returns no output (both deleted)
- [ ] `grep 'testLoRARoundTripFractionalScale' Tests/TuberiaTests/LoRAIntegrationTests.swift` returns a match (must NOT have been deleted)
- [ ] `grep -c 'unknownChipStringReturnsUnknownGeneration\|unknownGenerationHasNoNeuralAccelerator' Tests/TuberiaTests/DeviceCapabilityTests.swift` returns 2
- [ ] TuberiaTests compiles and passes: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaTests`

---

## Work Unit: TuberiaGPUTests

### Sortie 6: Add LoRA key convention tests

**Priority**: 7 — Highest priority in TuberiaGPUTests; Sortie 7 depends on target compiling. High risk: external key-matching conventions, complex string-matching logic.

**Entry criteria**:
- [ ] First sortie — no prerequisites
- [ ] `Tests/TuberiaGPUTests/LoRATests.swift` exists and the TuberiaGPUTests target compiles (verify: `test -f Tests/TuberiaGPUTests/LoRATests.swift`)

**Tasks**:
1. Read `Tests/TuberiaGPUTests/LoRATests.swift` in full to understand `LoRALoader.apply()` API, `ModuleParameters` construction pattern, and existing test style
2. Read `Sources/Tuberia/Pipeline/LoRALoader.swift` to understand how adapter weights are matched to base model keys — specifically: what key matching logic handles `lora_A.weight`/`lora_B.weight` suffixes, `lora_up`/`lora_down` suffixes, and `unet.` prefix normalization
3. Add `LoRAKeyConventionTests` suite to `Tests/TuberiaGPUTests/LoRATests.swift` with 4 tests:
   - `loraAWeightDotSuffix`: base key `"layer.weight"`, adapter keys `"layer.lora_A.weight"` / `"layer.lora_B.weight"` (HuggingFace-style) — verify merged result differs from base by the expected LoRA delta
   - `loraUpDownSuffix`: adapter keys `"layer.lora_up"` / `"layer.lora_down"` (diffusers-style) — verify merge produces the correct weighted sum
   - `unetPrefixIsStripped`: adapter key `"unet.layer.weight.lora_A"` normalizes to match base key `"layer.weight"` — verify weight is updated
   - `mixedConventionsSingleAdapter`: adapter whose keys mix `lora_A.weight`/`lora_B.weight` and `lora_up`/`lora_down` conventions across different layers — verify all matched layers are merged, no matched key silently skipped
4. All tests use synthetic `ModuleParameters`; no real adapter files; no network access; `eval()` before assertions

**Exit criteria**:
- [ ] `grep 'LoRAKeyConventionTests' Tests/TuberiaGPUTests/LoRATests.swift` returns a match
- [ ] `grep -c 'loraAWeightDotSuffix\|loraUpDownSuffix\|unetPrefixIsStripped\|mixedConventionsSingleAdapter' Tests/TuberiaGPUTests/LoRATests.swift` returns 4
- [ ] No `sleep` usage in new tests (verify: `grep 'sleep' Tests/TuberiaGPUTests/LoRATests.swift` — new tests only)
- [ ] TuberiaGPUTests compiles and all 4 new tests pass: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaGPUTests/LoRATests`

---

### Sortie 7: Add PipelineError thrown-condition tests

**Priority**: 3 — Depends on Sortie 6 (target must compile). Moderate risk.

**Decision**: Option B implemented — `DiffusionPipeline.generate()` now guards all three weighted segments at entry (`encoder.isLoaded`, `backbone.isLoaded`, `decoder.isLoaded`), throwing `PipelineError.missingComponent(role:)` for each. `PipelineError.missingComponent(role:)` already existed; no new error cases were needed.

**Entry criteria**:
- [ ] Sortie 6 exit criteria met — TuberiaGPUTests compiles
- [ ] `Tests/TuberiaGPUTests/ContractTests/PipelineAssemblyTests.swift` exists (290 lines — create separate file; do not add to this file)
- [ ] `Tests/TuberiaGPUTests/Mocks/MockTextEncoder.swift` exists (verify: `test -f Tests/TuberiaGPUTests/Mocks/MockTextEncoder.swift`)
- [ ] `Tests/TuberiaGPUTests/Mocks/MockDecoder.swift` exists (verify: `test -f Tests/TuberiaGPUTests/Mocks/MockDecoder.swift`)
- [ ] `Tests/TuberiaGPUTests/Mocks/MockWeightedSegment.swift` exists (verify: `test -f Tests/TuberiaGPUTests/Mocks/MockWeightedSegment.swift`)
- [ ] `grep 'guard encoder.isLoaded' Sources/Tuberia/Pipeline/DiffusionPipeline.swift` returns a match (Option B guard is present in production code)

**Tasks**:
1. Read `Tests/TuberiaGPUTests/ContractTests/PipelineAssemblyTests.swift` to understand pipeline contract test structure and which mock types are used to construct pipelines
2. Read `Tests/TuberiaGPUTests/Mocks/MockWeightedSegment.swift` to understand how `requiredKeys` triggers `PipelineError.weightLoadingFailed` when keys are missing from `ModuleParameters`
3. Read `Sources/Tuberia/Pipeline/PipelineError.swift` to confirm `missingComponent(role: String)` is the error case thrown by the `isLoaded` guards in `generate()`
4. Create `Tests/TuberiaGPUTests/ContractTests/PipelineErrorTests.swift` with `PipelineErrorTests` suite containing:
   - `generateWithUnloadedEncoderThrows`: construct a pipeline using fresh (never-`apply()`-called) `MockTextEncoder` plus loaded backbone/decoder mocks; call `generate()`; use `#expect(throws:)` to assert it throws `PipelineError.missingComponent` with `role == "encoder"` (REQ-TCG-050)
   - `generateWithUnloadedDecoderThrows`: same pattern but unloaded decoder mock; assert throws `PipelineError.missingComponent` with `role == "decoder"` (REQ-TCG-050)
   - `applyWeightsWithMissingKeyThrowsDescriptiveError`: construct `MockWeightedSegment(requiredKeys: ["model.weight"])` with empty `ModuleParameters`; call `apply(weights:)`; assert throws `PipelineError.weightLoadingFailed`; verify the caught error's `reason` string contains `"model.weight"` (REQ-TCG-050)
5. Use `#expect(throws:)` pattern from Swift Testing (not `XCTAssertThrows`); do NOT write new mock types — use existing mocks only; a freshly constructed mock that has never had `apply(weights:)` called starts with `isLoaded == false`

**Exit criteria**:
- [ ] `grep -r 'PipelineErrorTests' Tests/TuberiaGPUTests/` returns a match
- [ ] `grep -r 'applyWeightsWithMissingKeyThrowsDescriptiveError' Tests/TuberiaGPUTests/` returns a match
- [ ] Tests use `#expect(throws:)` pattern (verify: `grep -r '#expect.*throws\|#require.*throws' Tests/TuberiaGPUTests/ContractTests/` returns ≥ 1 match)
- [ ] `grep -r '#expect.*throws\|#require.*throws' Tests/TuberiaGPUTests/ContractTests/PipelineErrorTests.swift` returns ≥ 1 match
- [ ] TuberiaGPUTests compiles and all new tests pass: `xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS,arch=arm64' -only-testing TuberiaGPUTests`

---

## Parallelism Structure

**Critical Path**: TuberiaCatalogTests (Sortie 1 → 2 → 3 → 4 = 4 sorties, all sequential within work unit)

**Parallel Execution Groups**:
- **Group 1** (can start immediately):
  - TuberiaCatalogTests: Sortie 1 (Agent 1 — supervising)
  - TuberiaTests: Sortie 5 (Agent 2 — sub-agent, no build needed until exit criteria)
  - TuberiaGPUTests: Sortie 6 (Agent 3 — sub-agent, no build needed until exit criteria)
- **Group 2** (after Group 1 completes within each work unit):
  - TuberiaCatalogTests: Sorties 2 → 3 → 4 (Agent 1 — all builds via supervising agent)
  - TuberiaGPUTests: Sortie 7 (Agent 1 or Agent 3 — after Sortie 6)

**Agent Constraints**:
- **Supervising agent**: Handles all sorties with `xcodebuild test` exit criteria
- **Sub-agents (up to 2)**: Can handle research and file-writing phases of Sorties 5 and 6, but any sortie with a build exit criterion must return control to the supervising agent for the final build step

**Missed Opportunities**:
- Sorties 2, 3, 4 are all independent after Sortie 1, but are serialized within the TuberiaCatalogTests work unit. Could be separated into 3 work units for full parallelism — not recommended given the added orchestration overhead for 3 small sorties.

---

## Open Questions & Missing Documentation

### Resolved Items

| Sortie | Issue Type | Description | Resolution |
|--------|-----------|-------------|------------|
| 7 | ~~Open question~~ RESOLVED | **`generateWithUnloadedEncoderThrows` / `generateWithUnloadedDecoderThrows`**: `generate()` had no pre-flight `isLoaded` check. | **Option B chosen**: Added `guard encoder.isLoaded`, `guard backbone.isLoaded`, `guard decoder.isLoaded` at entry to `DiffusionPipeline.generate()`, each throwing `PipelineError.missingComponent(role:)`. No new error cases needed — `missingComponent(role:)` already existed. Production change committed to `Sources/Tuberia/Pipeline/DiffusionPipeline.swift`. |

### Auto-Fixed Issues

| Sortie | Issue Type | Description | Fix Applied |
|--------|-----------|-------------|-------------|
| 5 | Wrong API name | Plan used `hasNeuralEngine` — the actual property is `hasNeuralAccelerators` (`Sources/Tuberia/Infrastructure/DeviceCapability.swift:21`) | Fixed in Sortie 5 Task 4 and throughout |
| 5 | Vague path | "DeviceCapability.swift (or wherever chip parsing lives)" | Fixed to exact path: `Sources/Tuberia/Infrastructure/DeviceCapability.swift` |
| 7 | Vague condition | "if file is already large, create PipelineErrorTests.swift" — `PipelineAssemblyTests.swift` is 290 lines | Resolved: always create `Tests/TuberiaGPUTests/ContractTests/PipelineErrorTests.swift` |
| 7 | Non-verifiable criterion | "No real weights loaded; all mock objects in unloaded state" — not machine-checkable | Replaced with grep on `#expect(throws:)` pattern count |
| 2 | Missing criterion | No exit check for `eval()` pattern compliance (global acceptance criterion 2) | Added: `grep -c 'eval()'` check to Sortie 2 exit criteria |
| 2 | Missing criterion | No exit check for MARK comment block presence (global acceptance criterion 4) | Added: `grep '// MARK:'` check to Sortie 2 exit criteria |
| 3 | Missing criterion | No exit check for `eval()` pattern compliance (global acceptance criterion 2) | Added: scoped grep over new suite section in Sortie 3 exit criteria |
| 4 | Missing criterion | No exit check for MARK comment block presence (global acceptance criterion 4) | Added: `grep '// MARK:'` check to Sortie 4 exit criteria |

### Low-Impact Items (no sortie action needed)

| Scope | Issue Type | Description | Resolution |
|-------|-----------|-------------|------------|
| General | Manual verification only | Global acceptance criterion 5 requires a coverage report before and after deletion (Part 1) to confirm no coverage reduction. No `xccov` baseline exists in the repo. | Proxy: Sortie 1 exit criteria already verify all superset tests (`applyAndUnloadLifecycle`, `forwardPassShape8x8`, `outputShapeMatchesInput`) still pass after deletion, which guarantees no coverage loss. Human reviewer should confirm this proxy is acceptable. |

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 3 |
| Total sorties | 7 (was 6; original Sortie 3 split into Sortie 3 + Sortie 4; T5 strengthening merged into Sortie 1) |
| Dependency structure | 1 layer — all work units parallel |
| Requirements covered | 14 (REQ-TCG-001–006, 010–011, 020, 030, 040, 050, 060–061) |
| Tests removed | 6 redundant tests (REQ-TCG-001–006) |
| New test files | 3 (DPMSolverSchedulerTests, FlowMatchEulerSchedulerTests, ImageRendererUnitTests) |
| Suites added to existing files | 3 (SDXLVAEDecoderTensorTransformTests, LoRAKeyConventionTests, PipelineErrorTests) |
| Tests strengthened | 1 (applyWeightsDoesNotCrash) |
| Tests added to existing suites | 2 (DeviceCapabilityTests) |
| Maximum parallelism | 3 agents simultaneously (all WUs Layer 1) |
| Blocking issues | 0 — all open questions resolved |
