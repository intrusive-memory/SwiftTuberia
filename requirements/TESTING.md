# SwiftTubería — Testing Strategy

**Parent**: [`REQUIREMENTS.md`](../REQUIREMENTS.md) — SwiftTubería Overview
**Scope**: Testing requirements for both the `Tubería` and `TuberíaCatalog` targets.

---

## Component Tests (per pipe segment)

Each shared component is tested in isolation:
- **T5XXLEncoder**: known prompt → expected embedding shape and values
- **SDXLVAEDecoder**: known latent tensor → expected pixel output (PSNR vs reference)
- **DPMSolverScheduler**: synthetic noise predictions → expected denoising trajectory
- **ImageRenderer**: known pixel array → valid CGImage with correct dimensions
- **AudioRenderer**: known sample array → valid WAV with correct format

---

## Contract Tests (pipe compatibility)

Validate that outlet shapes match inlet expectations:
- T5XXLEncoder outlet dim (4096) matches PixArtDiT conditioning inlet (4096)
- T5XXLEncoder maxSequenceLength (120) matches PixArtDiT expectedMaxSequenceLength (120)
- SDXLVAEDecoder inlet channels (4) matches PixArt backbone output channels (4)
- Pipeline assembly with mismatched embedding dim fails with `PipelineError.incompatibleComponents`
- Pipeline assembly with mismatched sequence length fails with `PipelineError.incompatibleComponents`
- Recipe with `supportsImageToImage = true` and non-`BidirectionalDecoder` decoder fails at assembly
- `WeightedSegment.apply(weights:)` with missing keys throws clear error
- `WeightedSegment.apply(weights:)` with correct keys succeeds and sets `isLoaded = true`
- `WeightLoader.load()` applies `keyMapping`, `tensorTransform`, and `QuantizationConfig` correctly (synthetic safetensors with known key names and tensor shapes)

---

## Integration Tests

Full pipeline smoke tests per model (provided by model plugin test suites):
- PixArt recipe: prompt → CGImage (correct dimensions, non-zero pixels)

**Seed reproducibility thresholds**:
- Same device, same seed → PSNR > 40 dB between runs ("visually identical")
- Cross-platform (macOS vs iPadOS, different M-series) → PSNR > 30 dB (MLX float-point order may differ across GPU architectures)
- Byte-for-byte reproduction is NOT guaranteed and NOT required

**Weight conversion thresholds**:
- Converted weights (PyTorch → MLX safetensors) must produce output within PSNR > 30 dB of PyTorch reference
- Per-layer validation: investigate if any single layer drops below 25 dB (even if end-to-end passes)

---

## Infrastructure Tests

- Acervo integration: component access via handles, download orchestration (tested in SwiftAcervo; Pipeline tests verify the integration seam)
- Weight Loader: safetensors parsing, key remapping via `KeyMapping`, tensor transforms via `TensorTransform`, quantization via `QuantizationConfig`, delivery via `ModuleParameters` → `apply(weights:)`
- Memory Manager: device detection, budget enforcement, phase coordination

---

## Coverage and CI Stability Requirements

- All new code must achieve **≥90% line coverage** in unit tests. Coverage is measured per-target (`Tubería` and `TuberíaCatalog` separately) and enforced in CI.
- **No timed tests**: Tests must not use `sleep()`, `Task.sleep()`, `Thread.sleep()`, fixed-duration `XCTestExpectation` timeouts, or any wall-clock assertions. All asynchronous behavior must be validated via deterministic synchronization (`async`/`await`, `AsyncStream`, fulfilled expectations with immediate triggers).
- **No environment-dependent tests**: Protocol conformance tests, pipeline assembly/validation tests, scheduler math tests, and renderer data-transformation tests must use synthetic inputs and mock components — no real model weights, network access, or GPU required. Tests requiring downloaded models and GPU compute (e.g., T5XXLEncoder encoding, SDXLVAEDecoder PSNR checks) are integration tests and must be clearly separated (separate test target or `#if INTEGRATION_TESTS` gate).
- **Flaky tests are test failures**: A test that passes intermittently is treated as a failing test until fixed. CI must not use retry-on-failure to mask flakiness.
