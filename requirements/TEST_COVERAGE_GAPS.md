# SwiftTuberia — Test Coverage Gap Requirements

**Parent**: [`REQUIREMENTS.md`](../REQUIREMENTS.md) — SwiftTuberia Overview
**Scope**: Remediation requirements derived from the 2026-04-08 test suite evaluation. Covers both
redundant tests to remove and new tests to write. All new tests must comply with the CI stability
rules in [`TESTING.md`](TESTING.md) (no timed tests, no real weights, no network access).

---

## Part 1 — Redundant Tests to Remove

These tests are strict subsets of other tests in the same suite. Removing them reduces noise without
losing any coverage.

### REQ-TCG-001: Remove `applyWeightsSetsIsLoaded` from `T5ApplyWeightsTests`

**File**: `Tests/TuberiaCatalogTests/T5KeyMappingTests.swift`
**Test**: `T5ApplyWeightsTests.applyWeightsSetsIsLoaded`
**Reason**: `applyAndUnloadLifecycle` in the same suite checks `false → true → false`, which is a
strict superset. This test asserts only `false → true`.
**Acceptance**: Test deleted; `applyAndUnloadLifecycle` still passes.

---

### REQ-TCG-002: Remove `unloadResetsIsLoaded` from `T5ApplyWeightsTests`

**File**: `Tests/TuberiaCatalogTests/T5KeyMappingTests.swift`
**Test**: `T5ApplyWeightsTests.unloadResetsIsLoaded`
**Reason**: Same as REQ-TCG-001 — covered by `applyAndUnloadLifecycle`.
**Acceptance**: Test deleted; `applyAndUnloadLifecycle` still passes.

---

### REQ-TCG-003: Remove `outputHas3Channels` from `SDXLVAEDecoderModelTests`

**File**: `Tests/TuberiaCatalogTests/SDXLVAEModelTests.swift`
**Test**: `SDXLVAEDecoderModelTests.outputHas3Channels`
**Reason**: `forwardPassShape8x8` asserts `out.shape == [1, 64, 64, 3]`, which already implies
`out.shape[3] == 3`.
**Acceptance**: Test deleted; `forwardPassShape8x8` still passes.

---

### REQ-TCG-004: Remove `residualConnectionShape` from `AttentionBlockTests`

**File**: `Tests/TuberiaCatalogTests/SDXLVAEModelTests.swift`
**Test**: `AttentionBlockTests.residualConnectionShape`
**Reason**: `outputShapeMatchesInput` already verifies that output shape equals input shape for the
same channel count. The only difference here is using `ones * 2.0` vs `zeros` as input, which does
not test any new code path.
**Acceptance**: Test deleted; `outputShapeMatchesInput` still passes.

---

### REQ-TCG-005: Remove `testLoRAScaleZeroNoChange` from `LoRAIntegrationTests`

**File**: `Tests/TuberiaTests/LoRAIntegrationTests.swift`
**Test**: `LoRAIntegrationTests.testLoRAScaleZeroNoChange`
**Reason**: `LoRATests.loraScaleZero` in the GPU suite tests the identical behavior with a more
rigorous `diff.max()` assertion on a 4×4 weight. Keeping both only creates maintenance surface.
**Acceptance**: Test deleted; `LoRATests.loraScaleZero` (GPU suite) still passes.

---

### REQ-TCG-006: Remove `testLoRAFractionalScale` from `LoRAIntegrationTests`

**File**: `Tests/TuberiaTests/LoRAIntegrationTests.swift`
**Test**: `LoRAIntegrationTests.testLoRAFractionalScale`
**Reason**: `LoRATests.loraScaleHalf` in the GPU suite tests scale=0.5 with the same mathematical
expectation. `LoRAIntegrationTests.testLoRARoundTripFractionalScale` (scale=0.75 round-trip) is
kept because it has no GPU-suite equivalent.
**Acceptance**: Test deleted; `LoRATests.loraScaleHalf` (GPU suite) still passes.

---

## Part 2 — New Tests: Scheduler Unit Tests

Both schedulers have GPU integration test files but no CPU-level unit tests for their core math.

### REQ-TCG-010: `DPMSolverScheduler` CPU unit tests

**Target**: `TuberiaCatalogTests`
**New file**: `Tests/TuberiaCatalogTests/DPMSolverSchedulerTests.swift`

**Suite: `DPMSolverSchedulerConfigurationTests`**

| Test | Requirement |
|---|---|
| `configureProducesCorrectStepCount` | After `configure(numSteps:)`, `plan.timesteps.count == numSteps` |
| `linearBetaScheduleIsDifferentFromCosine` | `configure` with `.linear` vs `.cosine` produces different sigma sequences |
| `sigmasAreMonotonicallyDecreasing` | Each sigma in the schedule is strictly less than the previous |
| `firstSigmaIsLargest` | `plan.sigmas.first` > all subsequent sigmas |
| `lastSigmaApproachesZero` | `plan.sigmas.last` < 0.01 |

**Suite: `DPMSolverStepTests`**

| Test | Requirement |
|---|---|
| `stepReducesNoiseMagnitude` | Given a noisy latent and a noise prediction, `step()` output has lower L2 norm than input |
| `stepPreservesShape` | `step()` output shape matches input latent shape |
| `addNoiseIncreasesL2Norm` | `addNoise(latents:noise:timestep:)` output has higher L2 norm than clean input |
| `addNoisePreservesShape` | Shape contract: same shape as input latent |

**Constraints**: All tests use synthetic `MLXArray` inputs with known shapes. No real model weights.
No network access. Must pass without GPU (CPU Metal backend acceptable).

---

### REQ-TCG-011: `FlowMatchEulerScheduler` CPU unit tests

**Target**: `TuberiaCatalogTests`
**New file**: `Tests/TuberiaCatalogTests/FlowMatchEulerSchedulerTests.swift`

**Suite: `FlowMatchEulerConfigurationTests`**

| Test | Requirement |
|---|---|
| `configureProducesCorrectStepCount` | After `configure(numSteps:)`, `plan.timesteps.count == numSteps` |
| `sigmasSpanFullRange` | First sigma ≈ 1.0, last sigma ≈ 0.0 (within 0.05) |
| `sigmasAreMonotonicallyDecreasing` | Strictly decreasing sequence |
| `differentStepCountsProduceDifferentSchedules` | 10-step vs 20-step schedules are not equal |

**Suite: `FlowMatchEulerStepTests`**

| Test | Requirement |
|---|---|
| `stepPreservesShape` | Output shape matches input latent shape |
| `stepWithZeroVelocityIsIdentity` | With a zero velocity prediction, step output equals the input (no movement) |
| `addNoisePreservesShape` | Shape contract for img2img noise addition |

**Constraints**: Same as REQ-TCG-010.

---

## Part 3 — New Tests: Renderer Unit Tests

### REQ-TCG-020: `ImageRenderer` CPU unit tests

**Target**: `TuberiaCatalogTests`
**New file**: `Tests/TuberiaCatalogTests/ImageRendererUnitTests.swift`

**Suite: `ImageRendererConversionTests`**

| Test | Requirement |
|---|---|
| `floatZeroMapsToUInt8Zero` | An all-zeros `[1, 4, 4, 3]` input produces pixels with R=G=B=0 in the CGImage |
| `floatOneMapsToUInt8_255` | An all-ones `[1, 4, 4, 3]` input produces pixels with R=G=B=255 |
| `outOfRangeValuesAreClamped` | Values > 1.0 clamp to 255; values < 0.0 clamp to 0. No crash |
| `outputImageHasCorrectDimensions` | `[1, H, W, 3]` input produces a `CGImage` with `width == W` and `height == H` |
| `outputBitDepthIs8` | `CGImage.bitsPerComponent == 8` |
| `outputColorSpaceisSRGB` | `CGImage.colorSpace` is sRGB or compatible |

**Constraints**: Uses `CoreGraphics` to sample rendered pixels. No GPU required.

---

## Part 4 — New Tests: Weight Loading

### REQ-TCG-030: `SDXLVAEDecoder.tensorTransform` unit tests

**Target**: `TuberiaCatalogTests`
**Add suite to**: `Tests/TuberiaCatalogTests/SDXLVAEDecoderTests.swift`

**Suite: `SDXLVAEDecoderTensorTransformTests`**

| Test | Requirement |
|---|---|
| `convWeightIsTransposed` | A 4D array of shape `[out, in, kH, kW]` passed through `tensorTransform` for a `.weight` key containing `"conv"` in the key path produces shape `[out, kH, kW, in]` |
| `biasPassesThroughUnchanged` | A 1D bias array shape is unchanged by `tensorTransform` for a `.bias` key |
| `nonConvWeightPassesThroughUnchanged` | A 2D weight (e.g. layer norm) for a key not containing `"conv"` passes through unchanged |
| `transpositionIsCorrect` | A known 4D tensor with distinct values in each position has its axes reordered exactly as `[0, 2, 3, 1]` (i.e. `[out,in,kH,kW]` → `[out,kH,kW,in]`) |

**Constraints**: All synthetic tensors. No real weights. No GPU required.

---

## Part 5 — New Tests: LoRA Key Convention Coverage

### REQ-TCG-040: Alternative LoRA naming convention tests

**Target**: `TuberiaGPUTests`
**Add suite to**: `Tests/TuberiaGPUTests/LoRATests.swift`

**Suite: `LoRAKeyConventionTests`**

| Test | Requirement |
|---|---|
| `loraAWeightDotSuffix` | Keys using `lora_A.weight` / `lora_B.weight` suffix (HuggingFace style) are matched and merged correctly |
| `loraUpDownSuffix` | Keys using `lora_up` / `lora_down` suffix (diffusers style) are matched and merged correctly |
| `unetPrefixIsStripped` | Keys prefixed with `unet.` (diffusers safetensors convention) are normalized and matched to base model keys |
| `mixedConventionsSingleAdapter` | An adapter using a mix of suffix conventions for different layers still merges all matched keys and skips none silently |

**Constraints**: All synthetic `ModuleParameters`. No real adapter files. No network access.

---

## Part 6 — New Tests: Error Conditions

### REQ-TCG-050: `PipelineError` thrown-condition tests

**Target**: `TuberiaGPUTests`
**Add suite to**: `Tests/TuberiaGPUTests/ContractTests/PipelineAssemblyTests.swift` or new file

**Suite: `PipelineErrorTests`**

| Test | Requirement |
|---|---|
| `generateWithUnloadedEncoderThrows` | Calling `generate()` when the encoder has not been loaded (i.e. `isLoaded == false`) throws `PipelineError.encoderNotLoaded` (or equivalent) |
| `generateWithUnloadedDecoderThrows` | Calling `generate()` when the decoder has not been loaded throws `PipelineError.decoderNotLoaded` (or equivalent) |
| `applyWeightsWithMissingKeyThrowsDescriptiveError` | `WeightedSegment.apply(weights:)` with a `ModuleParameters` that is missing required keys throws an error; the error message includes the missing key name |

**Constraints**: Use existing mocks (`MockTextEncoder`, `MockDecoder`, etc.). Force unloaded state
by not calling `apply(weights:)`. No real weights or GPU compute required beyond Metal backend.

---

## Part 7 — Existing Test Improvements

### REQ-TCG-060: Strengthen `applyWeightsDoesNotCrash`

**File**: `Tests/TuberiaCatalogTests/T5KeyMappingTests.swift`
**Test**: `T5ApplyWeightsTests.applyWeightsDoesNotCrash`
**Current state**: Asserts only that no exception is thrown. Provides no evidence that any weight
was actually updated.
**Required change**: After `apply(weights:)`, sample at least one known parameter from the
transformer (e.g. `embedding.weight`) and assert it is not the same as its default initialization
value. This confirms loading actually mutated model state.
**Acceptance**: Test passes and includes at least one assertion beyond "no crash".

---

### REQ-TCG-061: Add unknown-chip coverage to `DeviceCapabilityTests`

**File**: `Tests/TuberiaTests/DeviceCapabilityTests.swift`

| Test | Requirement |
|---|---|
| `unknownChipStringReturnsUnknownGeneration` | Passing an unrecognized chip string (e.g. `"apple m99"`, `"intel core i9"`) returns `.unknown` from chip parsing |
| `unknownGenerationHasNoNeuralAccelerator` | `DeviceCapability` constructed from an `.unknown` generation reports `hasNeuralEngine == false` |

---

## Acceptance Criteria (all items)

1. All new test suites compile and run in the non-GPU targets (`TuberiaCatalogTests`,
   `TuberiaTests`) without requiring Metal or real model files.
2. Tests that touch `MLXArray` follow the `eval()` pattern before asserting on values.
3. No test uses `sleep()`, `Thread.sleep()`, `Task.sleep()`, or wall-clock assertions.
4. Each new test file includes a MARK comment block at the top documenting what crash or gap it
   regresses, matching the style in `SDXLVAEModelTests.swift`.
5. Removed tests (Part 1) do not cause any reduction in line coverage as measured by the remaining
   tests — verified by a coverage report before and after deletion.
