# FIX-45 — VAE decode transient allocation spike

**Issue:** [intrusive-memory/SwiftTuberia#45](https://github.com/intrusive-memory/SwiftTuberia/issues/45) — "[P0] VAE decode transient memory spike: no cache flush before decode, float32 decode, no tiled decode (cross-platform)"
**Reported from:** [intrusive-memory/Vinetas#83](https://github.com/intrusive-memory/Vinetas/issues/83)
**Blocks:** [intrusive-memory/Vinetas#92](https://github.com/intrusive-memory/Vinetas/issues/92) (macOS 4K resolution) — at 4K the decode transient grows ~16× vs. 1024², so the uncapped, float32, cache-polluted decode path is the gating defect for high-resolution output.

> Status: research/spec only. No source was modified, nothing was built. Line numbers below were re-verified against the current `development` checkout (the issue text was written against a slightly older revision; the corrected numbers are noted inline where they drifted).

## Problem statement

The denoise loop operates on small latents (`[1, H/8, W/8, 4]`), but the final SDXL VAE decode expands them to a full-resolution pixel tensor (`[1, H, W, 3]`) and, along the way, materializes a chain of large convolution activations at 512 / 512 / 256 / 128 channels through three 2× upsample stages. For 1024×1024 this transient is hundreds of MB to >1 GB, and it lands *on top of* the resident model weights plus MLX's still-pooled denoise/DiT buffers. Peak RSS = weights + pooled denoise buffers + VAE activation chain, all live simultaneously. That is an instant OOM on iOS and the peak-RAM moment on macOS. Every defect below is platform-agnostic — the fix must **not** be `#if os(iOS)`-gated.

## Root cause — the memory-stacking mechanics

Three independent effects compound at the decode boundary:

1. **MLX's GPU buffer cache is never flushed before decode.** The denoise loop calls `eval(latents)` once per step (`DiffusionPipeline.swift:1282`), so per-step DiT activations correctly drop out of Swift scope — but MLX does not *return* those freed buffers to the OS; it retains them in its internal allocator cache for reuse. `MemoryManager.clearGPUCache()` (which calls `MLX.Memory.clearCache()`, `MemoryManager.swift:131-133`) is the only thing that drains that cache, and it is called **only** from `unloadModels()` (`DiffusionPipeline.swift:675`), never inside `generate()`. So when `decoder.decode(latents)` runs (`DiffusionPipeline.swift:1331`), the cache from the denoise phase is still resident and the decode allocations stack on top of it instead of reusing that freed space.

2. **The entire decode runs in float32.** The CFG/no-CFG branches both `.asType(.float32)` the prediction before the scheduler step (`DiffusionPipeline.swift:1162-1164` and `1226`), so `latents` is float32 leaving the loop. `SDXLVAEDecoder.decode` only scales and `eval`s the latents (`SDXLVAEDecoder.swift:55-62`) — it never down-casts. `SDXLVAEDecoderModel.callAsFunction` (`SDXLVAEModel.swift:393-421`) does no internal `asType` either (there is no `asType` anywhere in `SDXLVAEModel.swift`). MLX up-promotes the fp16 conv weights to match the float32 input, so **every** activation buffer in the decode chain is float32 — 2× the bytes of an fp16 decode. The output is clipped to 8-bit in `ImageRenderer` anyway (`ImageRenderer.swift:37-38`: `clip(...)` then `.asType(.uint8)`), so the float32 precision is thrown away at the very end.

3. **The decode is a single monolithic pass.** `SDXLVAEDecoderModel.callAsFunction` decodes the whole latent in one forward pass. The three `Upsample2D` blocks (`SDXLVAEModel.swift:179-198`, driven from `VAEUpBlock.callAsFunction` `:261-276` and the up-block loop `:409-413`) each double H and W, producing the largest intermediates (e.g. for 1024² output the post-upsample tensors reach full 1024×1024 spatial extent at 512/256 channels). There is no tile/chunk path, so the transient scales linearly with output pixel count and is unbounded by anything except the consumer lowering `width × height` (`DiffusionPipeline.swift:921-922`, the `request.height / 8` / `request.width / 8` latent sizing).

Secondary: **hot-path `print(...)` on the decode path.** `SDXLVAEModel.swift` ships unconditional `print` calls at `:222,225,228,231` (`VAEMidBlock`), `:262,267,270,272` (`VAEUpBlock`), and `:394,398,402,406,412` (`SDXLVAEDecoderModel`). These run in release builds, on the hottest path, once per block per decode.

---

## Fixes, ordered by leverage / risk

### Fix 1 — Flush the MLX cache between denoise and decode (highest leverage, smallest, lowest risk)

**File / function:** `DiffusionPipeline.generate(...)`, between the end of the denoising loop (`DiffusionPipeline.swift:1312`) and `--- Step 5: Decode latents ---` (`:1314-1315`).

**Change.** Drain the GPU buffer cache after the final `eval(latents)` of the loop, before `decoder.decode(latents)`.

Before (`:1312-1331`):
```swift
        }
      }

      // --- Step 5: Decode latents ---
      progress(.decoding)
      let decoderScalingFactor = decoder.scalingFactor
      ...
      decodedOutput = try decoder.decode(latents)
```

After:
```swift
        }
      }

      // Drain MLX's GPU buffer cache before the VAE decode. The denoise loop
      // eval()s `latents` each step so per-step DiT activations leave scope,
      // but MLX retains the freed buffers in its allocator cache. Without this
      // flush the decode transient stacks on top of the pooled denoise buffers
      // instead of reusing that freed space (#45). `latents` is already
      // materialized by the loop's final eval(), so clearing the cache cannot
      // discard live work.
      await MemoryManager.shared.clearGPUCache()

      // --- Step 5: Decode latents ---
      progress(.decoding)
      let decoderScalingFactor = decoder.scalingFactor
      ...
      decodedOutput = try decoder.decode(latents)
```

**Why it's safe.** `MLX.Memory.clearCache()` only frees *cached/unreferenced* buffers; it never evicts arrays still referenced by a live `MLXArray`. `latents` is held by a Swift local and was just `eval`'d (`:1282`), so it is concrete and retained — clearing the cache cannot corrupt it. `generate()` is `async` on the actor and `MemoryManager` is an actor, so the `await` is already idiomatic here (the existing call site in `unloadModels()` `:675` uses the same form). Worst case it costs a few ms of re-allocation when decode warms back up; best case it removes the entire pooled-denoise contribution to peak RSS.

**MLX API:** `MemoryManager.shared.clearGPUCache()` → `MLX.Memory.clearCache()` (already wrapped, no new MLX surface needed).

---

### Fix 2 — Decode in fp16 (~halves the decode transient)

**File / function:** `SDXLVAEDecoder.decode(_:)`, `SDXLVAEDecoder.swift:54-62`.

**Change.** Down-cast the scaled latents to fp16 at decode entry so the whole VAE activation chain runs in 16-bit. Cast *after* the scalar scale-multiply (which is cheap and fine in fp32) and *before* the `eval`, so the materialized input handed to the model is already fp16.

Before (`:54-62`):
```swift
    // Apply internal scaling: latents * (1.0 / scalingFactor)
    let scaledLatents = latents * (1.0 / configuration.scalingFactor)

    // Force evaluation before the VAE forward pass. ...
    eval(scaledLatents)
```

After:
```swift
    // Apply internal scaling: latents * (1.0 / scalingFactor)
    // Down-cast to fp16 for the decode. The denoise loop leaves `latents` in
    // float32 (scheduler math), but the VAE weights are fp16 and the final
    // output is clipped to 8-bit in ImageRenderer, so a float32 decode wastes
    // ~2× the activation memory for precision that is discarded anyway (#45).
    // The cast is gated by configuration.decodeDType so it can be reverted to
    // float32 without code changes if a quality regression is found.
    let scaledLatents =
      (latents * (1.0 / configuration.scalingFactor)).asType(configuration.decodeDType)

    // Force evaluation before the VAE forward pass. ...
    eval(scaledLatents)
```

Add to `SDXLVAEDecoderConfiguration` (`SDXLVAEDecoderConfiguration.swift`):
```swift
import MLX  // for DType
...
  /// Compute dtype for the VAE decode pass. Defaults to fp16 to halve the
  /// decode-time activation footprint (#45). Set to `.float32` to restore the
  /// original full-precision decode if a quality regression is observed.
  public let decodeDType: DType

  public init(
    componentId: String = "sdxl-vae-decoder-fp16",
    latentChannels: Int = 4,
    scalingFactor: Float = 0.13025,
    decodeDType: DType = .float16
  ) {
    self.componentId = componentId
    self.latentChannels = latentChannels
    self.scalingFactor = scalingFactor
    self.decodeDType = decodeDType
  }
```

**Why it's safe.** The VAE weights are already fp16 (`componentId: "sdxl-vae-decoder-fp16"`, `estimatedMemoryBytes` comment says "~160 MB for fp16 SDXL VAE", `SDXLVAEDecoder.swift:107`). Currently MLX *up-promotes* those weights to float32 to match the float32 input; casting the input to fp16 instead lets the convs run natively in the weights' own precision — no up-promotion, half the activation bytes. `denormalizePixels` (`SDXLVAEDecoder.swift:98-100`) and `ImageRenderer` `clip → *255 → uint8` (`ImageRenderer.swift:37-38`) both tolerate fp16 input. The output dtype of `decode` changes from float32 to fp16; `DecodedOutput.data` is just an `MLXArray` (`Decoder.swift:25-34`) with no dtype contract, and the renderer re-casts to uint8 regardless, so no downstream consumer breaks.

**Open choice — fp16 vs bf16.** fp16 has more mantissa bits (better for the small post-`denormalize` [−1,1] range) but a narrow exponent range; bf16 is the safer default if any intermediate activation overflows fp16's ~65504 max. Recommend **fp16 first** (matches the weights, best precision in-range) and fall back to `.bfloat16` via the config flag only if NaN/Inf or saturation shows up in the decode telemetry (Fix 4) or the validation images. Do **not** introduce a separate bf16 weight set — the cast is on activations/input, weights stay as shipped.

**MLX API:** `MLXArray.asType(_:)` with `DType.float16` / `.bfloat16`.

---

### Fix 3 — Tiled / chunked decode (the only true cap on the transient)

**File / function:** new tiled path in `SDXLVAEDecoderModel` (`SDXLVAEModel.swift`), opt-in via `SDXLVAEDecoder.decode` / configuration. This is the durable structural fix and the one that unblocks Vinetas#92 (4K) — Fixes 1 & 2 reduce the constant factor; only tiling bounds the transient *independently of output resolution*.

**Approach.** Split the **latent** into spatial tiles, decode each tile through the full VAE, and stitch the resulting pixel tiles back together. Because the VAE is fully convolutional (every block is `Conv2d` / `GroupNorm` / nearest-`Upsample`, see `SDXLVAEModel.swift:17-198`) the receptive field is finite, so a tile decoded with enough latent-space halo (overlap) is numerically near-identical to the same region from a full-frame decode — except at the very edges, which the overlap/blend hides.

Sketch (new method on `SDXLVAEDecoderModel`, called by `decode` when a tile size is configured):

```swift
/// Tiled decode: bounds the activation transient to ~one tile regardless of
/// output resolution. `latentTile` is the tile edge in *latent* pixels;
/// `latentOverlap` is the halo added on each side before decoding and cropped
/// off after (in latent pixels). The VAE upsamples 8×, so output strides are
/// latentStride * 8.
func decodeTiled(_ x: MLXArray, latentTile: Int, latentOverlap: Int) -> MLXArray {
  let (B, H, W, _) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
  let scale = 8
  let outH = H * scale, outW = W * scale
  var output = MLXArray.zeros([B, outH, outW, 3], dtype: x.dtype)
  // Optional accumulation buffers for feathered blending across seams.

  var ly = 0
  while ly < H {
    var lx = 0
    while lx < W {
      // 1. Compute the haloed latent window, clamped to [0, H/W].
      let y0 = max(ly - latentOverlap, 0)
      let x0 = max(lx - latentOverlap, 0)
      let y1 = min(ly + latentTile + latentOverlap, H)
      let x1 = min(lx + latentTile + latentOverlap, W)
      let tile = x[0..., y0..<y1, x0..<x1, 0...]

      // 2. Full VAE forward on just this tile (the existing single-pass body).
      var decoded = self.callAsFunction(tile)   // [B, (y1-y0)*8, (x1-x0)*8, 3]
      eval(decoded)

      // 3. Crop the halo back off in *pixel* space and write into `output`
      //    at (ly*8, lx*8). Feather the seam over the overlap region if a
      //    hard crop shows a visible edge (see blending note below).
      // ... slice `decoded` to the interior region and assign into `output`.

      MemoryManager... // (optional) drain cache between tiles to hold the cap

      lx += latentTile
    }
    ly += latentTile
  }
  return output
}
```

Wiring: add `decodeTileLatentSize: Int?` and `decodeTileLatentOverlap: Int` to `SDXLVAEDecoderConfiguration`; in `SDXLVAEDecoder.decode`, if `decodeTileLatentSize` is non-nil **and** the latent is larger than the tile, call `loadedModel.decodeTiled(...)` instead of `loadedModel(scaledLatents)` (`SDXLVAEDecoder.swift:72-83`). Default `nil` (no tiling) so existing behavior is bit-for-bit unchanged until a consumer opts in.

**Tiling parameters (starting point).**
- **Tile size:** 64 latent px (= 512 output px) is a reasonable first cap; 96 (=768) trades memory for fewer seams. Make it configurable.
- **Overlap/halo:** the VAE's effective receptive field is a handful of latent pixels (3×3 convs across ~7 conv layers per resolution, plus the mid-block attention which is *global* at the bottleneck). The global attention is the real subtlety — a tile sees only its own bottleneck context, so very large tiles are safer for global-coherence. Start with **8 latent px** overlap and tune up if seams appear.
- **Reassembly / seam handling:** simplest is a hard crop of the halo (decode the haloed tile, keep only the central `latentTile*8` pixel region). If hard-crop seams are visible, switch to **feathered blending**: keep the overlap pixels, weight each tile's contribution with a linear (or cosine) ramp across the overlap band, and sum into an accumulation buffer divided by a weight buffer. Feathering is the standard fix for conv-VAE tiling seams (it's what diffusers' `tiled_decode` does).

**Open questions (must be resolved during implementation).**
1. **Global mid-block attention** (`AttentionBlock`, `SDXLVAEModel.swift:105-171`) means a tile's bottleneck attends only over its own spatial extent. With small tiles this can shift low-frequency tone/contrast tile-to-tile. Mitigations: larger tiles, generous overlap + feather, or (heavier) decode the mid-block once at full latent res and tile only the upsample stages. Decide empirically; start with hard tiling + feather and inspect.
2. **Seam visibility threshold** — pick the overlap/feather width by eye on a high-frequency test image (text, fine texture) vs. a full-frame reference.
3. **Does tiling compose with fp16 (Fix 2)?** Yes — they're orthogonal; tiling bounds the count of live activations, fp16 halves each. Land Fix 2 first so the tiled path inherits the smaller dtype.
4. **Per-tile cache flush** — calling `clearGPUCache()` between tiles enforces the cap but costs realloc latency; measure whether it's needed or whether MLX reuses the tile buffer naturally.

**Why it's safe.** Default-off (config `nil`) means zero behavior change until opted into. The tiled path reuses the exact same `callAsFunction` body per tile, so per-tile correctness == current correctness; the only new risk is seam artifacts, addressed by overlap+feather and gated behind validation.

**MLX API:** array slicing (`x[0..., y0..<y1, x0..<x1, 0...]`), `MLXArray.zeros(_:dtype:)`, slice-assignment for stitching, `eval(_:)` per tile.

---

### Fix 4 — Telemetry / resident-memory sampling around decode

**Files:** `TuberiaTelemetryEvent.swift` (new event or extend existing), `MemoryManager.swift` (add a resident-memory sampler), `DiffusionPipeline.generate` decode block (`:1314-1356`).

**Current state.** `decoderDecodeStart` / `decoderDecodeComplete` events already exist (`TuberiaTelemetryEvent.swift:155-156`) and are emitted around the decode (`DiffusionPipeline.swift:1317-1356`), but they only carry tensor *stats* and duration — **no resident/peak memory**. `MemoryManager` tracks only registered *weight* bytes (`loadedComponentsMemory`, `:123-125`) and never samples actual RSS; it imports `MachO` (`:6`) but does not call `task_info`.

**Change.** Add a resident-footprint sampler to `MemoryManager` and a peak-memory field to the decode-complete telemetry.

In `MemoryManager.swift` (Darwin only):
```swift
#if canImport(Darwin)
/// Current resident footprint (phys_footprint) of this process, in bytes.
/// This is the number Jetsam/the memory limit actually watches, so it is the
/// right signal for "did the decode spike toward the cap?" (#45).
public var residentFootprint: UInt64 {
  var info = task_vm_info_data_t()
  var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size
    / MemoryLayout<natural_t>.size)
  let kr = withUnsafeMutablePointer(to: &info) {
    $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
      task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
    }
  }
  guard kr == KERN_SUCCESS else { return 0 }
  return UInt64(info.phys_footprint)
}
#endif
```
Extend `decoderDecodeComplete` (preferred — it's already wired) to carry `residentBytesBefore` / `residentBytesAfter`, or add a dedicated `case decoderMemorySample(residentBytes: UInt64, phase: String)`. In `DiffusionPipeline.generate`, sample `await MemoryManager.shared.residentFootprint` immediately before `decoder.decode` (`:1331`) and immediately after (`:1342`) and attach both to the completion event.

**Why it's safe.** Pure observation; no behavioral change. `task_info(TASK_VM_INFO)` is read-only and cheap (microseconds). Gate the sampling behind the existing `if let telemetry` guard (`:1343`) so it's zero-cost when telemetry is off, consistent with the rest of the pipeline's "zero-cost-when-off" discipline. Use `phys_footprint` (the Jetsam-watched number), not `resident_size`.

---

### Fix 5 — Remove hot-path `print(...)` (hygiene)

**File:** `SDXLVAEModel.swift`. Unconditional `print` calls on the decode path at `:222,225,228,231` (`VAEMidBlock`), `:262,267,270,272` (`VAEUpBlock`), `:394,398,402,406,412` (`SDXLVAEDecoderModel`). (The issue cited `222-233 / 262-272 / 394-413`; these are the exact lines within those ranges.)

**Change.** Delete them, or route through a compile-gated logger. Simplest: delete. If the shape traces are wanted for debugging, gate behind `#if DEBUG` or a `Tuberia` log channel, but they must not ship in release — they execute once per block per decode and force string interpolation of `.shape` on every call. Note: the `eval(h)` calls interleaved with these prints (e.g. `:224,227,230` and `:397,401,405,411`) are **load-bearing** (the surrounding comments document shapeless-tensor crashes under memory pressure) — keep the `eval`s, remove only the `print`s.

---

## Quality / correctness risks

- **fp16 decode quality (Fix 2).** Risk: banding or hue shift in smooth gradients, or fp16 overflow on an unusually large intermediate. The VAE works in the small post-scale range, so overflow is unlikely, but validate. **Validation:** decode a fixed seed/prompt at float32 (baseline) and fp16; compare PSNR/SSIM of the two PNGs (expect PSNR > ~45 dB / SSIM > 0.99) and eyeball gradients/skies. If a regression appears, flip `decodeDType` to `.bfloat16`, then to `.float32` (revert) — the config flag makes this a one-line change with no recompile of call sites.
- **Tiling seam artifacts (Fix 3).** Risk: visible grid seams or tile-to-tile tone shifts from the global mid-block attention. **Validation:** decode the same latent full-frame vs. tiled; diff the images, inspect the seam bands specifically, and test a high-frequency image (text/texture). Increase overlap and enable feather blending until the diff at seams is below visible threshold. Keep tiling **default-off** until it passes this bar.
- **Cache flush (Fix 1).** Negligible correctness risk; only a small latency cost. Confirm no double-flush regression interacts with `unloadModels()` (it won't — different lifecycle points).
- **Telemetry (Fix 4).** None functional; just verify `task_info` returns `KERN_SUCCESS` on device and the sampling stays behind the telemetry guard.

## Test plan

**Unit (CPU, deterministic) — `Tests/TuberiaCatalogTests/`:**
- `SDXLVAEDecoderTests`: assert `decode` returns the documented output shape/range with `decodeDType = .float16` (extend the existing placeholder-path tests so they don't need real weights). Assert `denormalizePixels` still produces [0,1] from fp16 input.
- New config test: `SDXLVAEDecoderConfiguration` default `decodeDType == .float16`, `decodeTileLatentSize == nil`; round-trip the new fields.
- `MemoryManagerTests` (`Tests/TuberiaTests/`): add a `residentFootprint > 0` smoke assertion (Darwin only).

**GPU / model-dependent (must run on Apple Silicon with real weights) — `Tests/TuberiaCatalogGPUTests/SDXLVAEDecoderTests.swift`:**
- fp16-vs-fp32 decode parity: decode a fixed latent both ways, assert PSNR/SSIM above threshold.
- Tiled-vs-full parity: decode a fixed latent full-frame and tiled, assert per-pixel max-diff (and specifically seam-band diff) below threshold.
- These are inherently GPU/MLX/Metal tests — keep them in the GPU test target (which already exists) so they only run where Metal is available; do not add them to the plain `TuberiaCatalogTests` target.

**Peak-memory measurement (the actual issue metric):**
- With Fix 4 in place, run a 1024×1024 generate with telemetry on and record `decoderDecodeComplete` resident-before/after. Compare four configurations: (baseline) → (Fix 1) → (Fix 1+2) → (Fix 1+2+3 tiled). Expect: Fix 1 removes the pooled-denoise overhang; Fix 2 ~halves the decode delta; Fix 3 makes the decode delta roughly flat vs. tile size instead of growing with resolution.
- For a hard external check, run the same generate under `xcrun xctrace` / Instruments Allocations or watch `phys_footprint` and confirm the peak drops. (This is a manual/perf measurement, **not** a CI-gated assertion — peak-RAM thresholds are too flaky for CI. Do not add a hard peak-memory assertion to the gating suite.)

## Sequencing

1. **Land Fix 1 first** (pre-decode `clearGPUCache()`). One line, lowest risk, highest immediate leverage, independently shippable. Ship it alone.
2. **Fix 5 (remove prints)** — trivial, independent, ship alongside or before anything else.
3. **Fix 4 (telemetry/memory sampling)** — independent; land early so it can *measure* Fixes 2 and 3.
4. **Fix 2 (fp16 decode)** — independent of 1/4/5; gate behind the config flag, validate quality, then ship.
5. **Fix 3 (tiled decode)** — last and largest. Build on top of Fix 2 (so tiles inherit fp16), keep **default-off**, validate seams, and enable for the high-resolution path that unblocks Vinetas#92.

Fixes 1, 2, 4, 5 are all independently shippable and additive; only Fix 3 has meaningful design surface and should be its own PR with the parity/seam tests attached.
