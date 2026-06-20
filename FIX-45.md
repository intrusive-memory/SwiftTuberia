# FIX-45 — VAE decode transient allocation spike (remaining work: tiled decode)

**Issue:** [intrusive-memory/SwiftTuberia#45](https://github.com/intrusive-memory/SwiftTuberia/issues/45) — "[P0] VAE decode transient memory spike"
**Blocks:** [intrusive-memory/Vinetas#92](https://github.com/intrusive-memory/Vinetas/issues/92) (macOS 4K resolution)

> Status (2026-06-19): **partially resolved.** The constant-factor reductions (cache flush, fp16 decode), the print cleanup, and the decode memory telemetry have all landed on `fix/45`. The one piece left is the only true *cap* on the transient — **correct tiled decode** — which is wired and default-OFF but not yet correct on real weights because of the VAE's global mid-block attention. This document now covers ONLY that remaining work.

## What already shipped (do not redo)

- **Fix 1 — pre-decode `clearGPUCache()`.** `DiffusionPipeline.generate()` now drains the MLX buffer cache between the denoise loop and `decoder.decode(latents)`. Removes the pooled-denoise overhang from peak RSS.
- **Fix 2 — fp16 decode.** `SDXLVAEDecoder.decode` casts the scaled latents to `configuration.decodeDType` (default `.float16`) before the forward pass, so the whole activation chain runs in 16-bit. Revert to `.float32` (or `.bfloat16`) via the config flag with no code change.
- **Fix 4 — decode memory telemetry.** `MemoryManager.residentFootprint` samples `phys_footprint` via `task_info(TASK_VM_INFO)`; `decoderDecodeComplete` now carries `residentBytesBefore` / `residentBytesAfter`, sampled around the decode behind the telemetry guard.
- **Fix 5 — hot-path `print` removal.** Diagnostic prints in `SDXLVAEModel.swift` (`VAEMidBlock`, `VAEUpBlock`, `SDXLVAEDecoderModel`) are gone; the load-bearing interleaved `eval(h)` calls are kept.

## Remaining work — Fix 3: tiled / chunked decode (correctness on real weights)

### Current state on `fix/45`
`SDXLVAEDecoderModel.decodeTiled(_:latentTile:latentOverlap:)` exists and is wired into `SDXLVAEDecoder.decode` via two config fields (`decodeTileLatentSize: Int?`, default `nil`; `decodeTileLatentOverlap: Int`, default 8). It splits the latent into spatial tiles, decodes each haloed tile through the full VAE, **hard-crops** the halo, and stitches the interior into a full-resolution canvas.

What is **proven correct** today (unit tests, synthetic CPU weights):
- Single-tile case (`tile >= latent`) is bit-identical to full-frame decode.
- The crop/stitch arithmetic is exact: on a shift-invariant (all-zero) model a multi-tile decode equals full-frame to max-diff 0.

What is **NOT yet correct** (the open problem):
- **Global mid-block attention.** `AttentionBlock` (`SDXLVAEModel.swift`) attends over the *entire* spatial extent of whatever tensor it receives. A tile therefore attends only over its own bottleneck context, not the full image. On real (non-zero) weights this shifts low-frequency tone/contrast tile-to-tile and a hard crop leaves visible seams. **Tiling is default-OFF for exactly this reason and must stay off until the items below are resolved.**

### Open problems to resolve before enabling tiling for production
1. **Global attention vs. tiling.** Pick one of:
   - decode the mid-block once at full latent resolution and tile only the upsample stages (heavier, but keeps global attention intact — likely the correct answer), **or**
   - use large tiles + generous overlap + feathered blending and accept a small low-frequency error, validated by eye.
2. **Feathered seam blending.** Only hard-crop is implemented. Add linear/cosine ramp blending across the overlap band (accumulation buffer / weight buffer), the standard diffusers `tiled_decode` approach, if hard-crop seams are visible.
3. **Tile/overlap tuning.** Choose `decodeTileLatentSize` / `decodeTileLatentOverlap` by inspecting a high-frequency reference (text/texture) full-frame vs. tiled.
4. **Per-tile cache flush.** Measure whether `clearGPUCache()` between tiles is needed to hold the cap, vs. the realloc latency it costs.

### Required tests before flipping the default on (GPU / real weights)
These need real checkpoint weights and Metal, so they belong in `Tests/TuberiaCatalogGPUTests/` and were **not** added in this PR (no cached weights available in the dev/CI environment):
- **fp16-vs-fp32 decode parity** on real weights: PSNR > ~45 dB / SSIM > 0.99. (Validates the already-shipped Fix 2 on a real model, not just shape.)
- **Tiled-vs-full parity** on real weights: per-pixel max-diff and seam-band diff below a visible threshold. This is the gate for enabling tiling.

Do **not** add a hard peak-RAM assertion to the CI-gated suite (too flaky). Peak-RAM is a manual Instruments / `phys_footprint` measurement using the Fix 4 telemetry.

## Why this is a separate, future PR
Fixes 1/2/4/5 are additive and independently shippable and are in this PR. Fix 3 has real design surface (the global-attention decision) and must ship with its own real-weights parity/seam tests. Shipping a seam-artifacting decode would be worse than no tiling, so the path stays opt-in and unverified-on-real-weights until that work is done.
