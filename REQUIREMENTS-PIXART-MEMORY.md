---
type: project
title: "SwiftTuberia — PixArt iOS Peak-Memory Reduction Requirements"
date: 2026-07-10
status: "ACTIVE — not yet implemented"
master_index: "/Users/stovak/Projects/REQUIREMENTS.md"
supersedes: "REQUIREMENTS.md REQ-PIPE-02 (memory gate) — see REQ-MEM-03 below"
companion: "../SwiftVinetas/REQUIREMENTS-PIXART-MEMORY.md"
---

# SwiftTuberia — PixArt iOS Peak-Memory Reduction Requirements

**Mission**: Cut the peak resident footprint of a PixArt-Sigma generation on iOS so it
stops OOM-killing (jetsam) on iPad, by making `DiffusionPipeline` free the text encoder
before the denoise loop — the same phased strategy `Flux2Pipeline` already uses — and by
correcting the iOS memory gate to watch the per-process jetsam budget.

**Problem statement**: FLUX.2 (larger on paper) generates on an iPad where PixArt (smaller)
fails. Root cause is *phased memory management*, not model size. `DiffusionPipeline.loadModels()`
loads encoder + backbone + decoder in one loop and keeps **all three resident** through the
entire denoise loop and VAE decode. The int4 T5-XXL text encoder is ~1.2 GB (72 % of the
weight footprint) yet is used only by the two `encode()` calls before denoising, and is never
unloaded. Peak weights ≈ 1.66 GB (T5 ~1.2 GB + PixArt DiT ~300 MB + SDXL VAE ~160 MB); freeing
the encoder after encode drops the compute-phase resident set to ≈ 460 MB.

**Reference implementation (parity target)**: `flux-2-swift-mlx` `Sources/Flux2Core/Pipeline/Flux2Pipeline.swift`
— `unloadTextEncoder()` (`:437`, called `:1318` before the transformer loads), per-phase MLX
cache ceilings, `peakMemory = 0` resets, and `clearCacheEveryNSteps` mid-loop clears (`:1580`).

**Hard constraint**: The T5-XXL weights are consumed packed-int4 by `quantizedMM` at forward
time. Dequantizing to fp16 inflates the encoder ~1.2 GB → ~9.4 GB and OOMs the iPad (documented
at `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift:343-348`). **The fix is to *unload* the
encoder, never to re-quantize or dequantize it.** `T5XXLEncoder.unload()` (`:386-390`) already
exists and correctly frees the packed weights; it is simply never called mid-flight.

---

## Status Snapshot (2026-07-10)

| # | Area | Status | Evidence / Target |
|---|---|---|---|
| 1 | Free T5 encoder before denoise (two-phase load) | ❌ TODO | REQ-MEM-01 — `DiffusionPipeline.loadModels()`/`generate()` |
| 2 | Periodic MLX cache clear inside the denoise loop | ❌ TODO | REQ-MEM-02 — `DiffusionPipeline` denoise loop `:1019-1312` |
| 3 | iOS gate uses per-process jetsam budget + phased peak | ❌ TODO | REQ-MEM-03 — `MemoryManager.availableMemory`, `loadModels` gate |
| 4 | Per-phase `physFootprint` telemetry on the PixArt path | ❌ TODO | REQ-MEM-04 — `TuberiaTelemetryEvent` + emission sites |

The status snapshot is authoritative. Sorties are atomic — complete in order; REQ-MEM-01 is
the load-bearing change and must land (and be profiled) before REQ-MEM-02/03/04.

---

## Outstanding Work (Ordered)

### REQ-MEM-01: Free the Text Encoder After Encoding, Before the Denoise Loop

**Priority**: 🔴 CRITICAL — this is the OOM fix. Everything else is complementary.

**Files**:
- `Sources/Tuberia/Pipeline/DiffusionPipeline.swift` — `loadModels(progress:)` (`:516`), the
  `weightedSegments` load loop (`:568-663`), `generate(request:)` (`:689`), the
  `encoder.isLoaded` guard (`:693`), the two `encode()` calls (`:830`, `:877`), the denoise
  loop (`:1019-1312`).
- `Sources/Tuberia/Infrastructure/MemoryManager.swift` — `clearGPUCache()` (`:157`),
  `registerLoaded`/`unregisterLoaded` (`:113-119`).
- `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` — `unload()` (`:386`), the int4
  constraint note (`:343`).

**Current state**: `loadModels()` loads all three segments (encoder, backbone, decoder) into
memory in one loop and never unloads. `generate()` hard-guards `encoder.isLoaded` (`:693`) for
the whole run, structurally forcing the encoder to stay resident. The docstring at `:510-513`
already *describes* two-phase loading ("Phase 1: load encoder, encode, unload; Phase 2:
backbone + decoder") — the implementation never did it.

**Work**:
1. Split loading into two phases. Options, in preference order:
   - **(a) Load-encode-unload-then-load-rest** inside `generate()`: keep `loadModels()` loading
     encoder + backbone + decoder as today *only* on high-memory tiers; on constrained tiers,
     load the encoder first, run `encode()`, call `encoder.unload()` +
     `MemoryManager.shared.clearGPUCache()`, then load backbone + decoder. Requires the pipeline
     to defer backbone/decoder load out of `loadModels()` into `generate()` for the phased path.
   - **(b) Unload-after-encode with eager load retained**: simplest delta — keep eager load, but
     immediately after the last `encode()` in `generate()` call `encoder.unload()` +
     `clearGPUCache()` and drop the `encoder.isLoaded` requirement for the denoise/decode phases.
     Peak-at-load is unchanged, but the *sustained* footprint through the 20-step loop falls by
     ~1.2 GB. This alone likely clears the jetsam kill because the spike during denoise/decode is
     what trips the cap; verify against the REQ-MEM-04 profile.
   Choose (b) first (smaller, safer); escalate to (a) only if load-time peak alone still OOMs.
2. Relax the `encoder.isLoaded` precondition in `generate()` (`:693`) so a nil/unloaded encoder
   is valid once conditioning has been computed. Guard only that conditioning exists.
3. On unload, call `MemoryManager.shared.unregisterLoaded(component:)` for the encoder so
   `loadedComponentsMemory` reflects reality.
4. Preserve packed-int4 — do not touch quantization. Add an inline comment pointing at
   `T5XXLEncoder.swift:343` so no future change reintroduces a dequantize.

**Exit**:
- A test drives encode → asserts `encoder.isLoaded == false` (or the encoder ref is nil) before
  the first `backbone.forward()`, and generation still completes with correct output.
- The REQ-MEM-04 / SwiftVinetas profiler shows resident footprint dropping ~1.2 GB at the
  `encode → denoise` boundary instead of staying flat at ~1.6 GB.

---

### REQ-MEM-02: Periodic MLX Cache Clear Inside the Denoise Loop

**Priority**: 🔵 HIGH — complementary; bounds per-step transient accumulation.

**Files**:
- `Sources/Tuberia/Pipeline/DiffusionPipeline.swift` — denoise loop (`:1019-1312`), the existing
  single pre-decode clear (`:1321`), per-step `eval(latents)` (`:1282`).

**Current state**: The MLX buffer cache is cleared exactly once, after the loop and before VAE
decode (`:1321`). There is no per-step clear, so transient buffers from each step can accumulate.
`Flux2Pipeline` clears every N steps (`:1580`) tuned by RAM tier.

**Work**:
1. Add a configurable `clearCacheEveryNSteps` (default derived from device tier via
   `DeviceCapability`; 0 = never, matching current behavior on high-memory tiers).
2. Inside the denoise loop, after `eval(latents)`, call `MemoryManager.shared.clearGPUCache()`
   when `(stepIndex + 1) % clearCacheEveryNSteps == 0`.
3. Default constrained/iOS tiers to a small N (e.g. 1–2); leave macOS high-RAM at 0 so
   throughput is not throttled.

**Exit**: A profiled iOS run shows a flatter (non-monotonically-rising) resident footprint
across the denoise loop. No correctness regression in existing PixArt output tests.

---

### REQ-MEM-03: Correct the iOS Memory Gate — Per-Process Jetsam Budget + Phased Peak

**Priority**: 🔵 HIGH — the current gate can pass on a big iPad while the process is still
jetsam-killed. Supersedes REQUIREMENTS.md REQ-PIPE-02's up-front `peakMemoryBytes` gate.

**Files**:
- `Sources/Tuberia/Infrastructure/MemoryManager.swift` — `availableMemory` (`:47-75`),
  `softCheck`/`hardValidate` (`:81-108`).
- `Sources/Tuberia/Pipeline/DiffusionPipeline.swift` — the memory gate call (`:523-529`), the
  `peakMemoryBytes` vs `phasedMemoryBytes` requirement computation (`:181-192`).

**Current state**: `availableMemory` uses system-wide Mach VM statistics (`host_statistics64`),
which on iOS reflects *device* free memory, not the *per-process* jetsam cap. The gate validates
against `peakMemoryBytes = encoderMem + backboneMem + decoderMem` (all three summed, `:184`).
`phasedMemoryBytes = max(encoder, backbone+decoder)` is already computed (`:188`) but unused.

**Work**:
1. On iOS, source the budget from `os_proc_available_memory()` (the number Jetsam enforces),
   not system VM stats. Keep the Mach VM path for macOS (no per-process cap there). Gate on the
   platform-correct value.
2. Once REQ-MEM-01 lands, gate `loadModels()` against `phasedMemoryBytes` (`:188`) rather than
   `peakMemoryBytes` (`:184`), because the phased design never holds all three resident.
3. Keep emitting `memoryGateChecked(requiredBytes:passed:)` with the value actually gated on so
   telemetry matches the decision.

**Exit**: A unit test stubbing the per-process budget below `phasedMemoryBytes` throws
`PipelineError.insufficientMemory`; a stub above it passes. On device, a run that previously
jetsam-killed either completes or fails the gate *cleanly* (no silent kill).

---

### REQ-MEM-04: Per-Phase `physFootprint` Telemetry on the PixArt Path

**Priority**: 🟢 MEDIUM — makes the pipeline self-profiling; closes the asymmetry with Flux2.

**Files**:
- `Sources/Tuberia/Telemetry/TuberiaTelemetryEvent.swift` — event definitions.
- `Sources/Tuberia/Pipeline/DiffusionPipeline.swift` — emission sites (weight load, encode,
  per-step or per-N-step, decode `:1342-1366`).
- Consumer encoding lives in the companion repo — see
  `../SwiftVinetas/REQUIREMENTS-PIXART-MEMORY.md` REQ-VIN-02.

**Current state**: The Tuberia event stream reports resident footprint only around VAE decode
(`decoderDecodeComplete` `residentBytesBefore/After`, `:1342-1366`). Flux2 emits `physFootprint`
on nearly every phase event via `Flux2MemoryFootprint.current()`.

**Work**:
1. Add an optional `physFootprint: UInt64?` field (sourced from `MemoryManager.residentFootprint`,
   `:135-151`) to the weight-load, encode, denoise-progress, and decode events.
2. Emit it at each phase boundary so a consumer can reconstruct the footprint-by-phase timeline
   without a separate sampler.
3. Keep the field optional so existing consumers compile unchanged.

**Exit**: A telemetry integration run yields a per-phase footprint series; the
`encode → denoise` drop from REQ-MEM-01 is visible in the trace, not just the external profiler.

---

## Cross-References

- Root mission spec (do **not** conflict): `docs/complete/REQUIREMENTS.md` (SwiftAcervo v2 integration, completed).
  REQ-MEM-03 supersedes its REQ-PIPE-02 memory-gate decision.
- Consumer requirements: `../SwiftVinetas/REQUIREMENTS-PIXART-MEMORY.md` (version-floor bump,
  telemetry-adapter encoding, profiling harness).
- Parity reference: `../flux-2-swift-mlx/Sources/Flux2Core/Pipeline/Flux2Pipeline.swift`.
- External profiler: `../SwiftVinetas/Sources/SwiftVinetas/Core/VinetasMemoryProfiler.swift`
  (`make profile-pixart-memory`) — use for the before/after baseline.

## Definition of Done (Master Acceptance)

1. On the target iPad, a PixArt-Sigma 512² generation that previously jetsam-killed completes.
2. Profiler / REQ-MEM-04 telemetry shows resident footprint dropping ~1.2 GB at the
   `encode → denoise` boundary (peak ~1.66 GB → sustained ~460 MB through the loop).
3. No dequantization of T5-XXL is introduced (packed int4 preserved).
4. Existing PixArt output-correctness tests still pass; macOS throughput is not regressed.
5. A new SwiftTuberia release is tagged and its version consumed downstream (SwiftVinetas
   REQ-VIN-01).

## History

| Date | Change |
|---|---|
| 2026-07-10 | File created. Root-caused PixArt iOS OOM to un-freed T5 encoder in `DiffusionPipeline`; four sorties (REQ-MEM-01..04) identified. REQ-MEM-01 is load-bearing. |
