# SwiftTuberia — Infrastructure Services

**Parent**: [`REQUIREMENTS.md`](../REQUIREMENTS.md) — SwiftTuberia Overview
**Scope**: Shared services available to all pipe segments and pipelines — model access via SwiftAcervo, WeightLoader, MemoryManager, DeviceCapability, and Progress Reporter. This document contains both the design rationale and the canonical Swift definitions.

---

## Model Access via SwiftAcervo

**SwiftTuberia does not have its own model registry.** All model discovery, download, caching, and file access is delegated to SwiftAcervo's Component Registry (see SwiftAcervo REQUIREMENTS.md).

Model plugins register their `ComponentDescriptor` entries with Acervo at import time. **Catalog components also self-register**: when `TuberiaCatalog` is imported, it registers Acervo descriptors for all shared components (T5-XXL, SDXL VAE, DPM-Solver weights if any, etc.) via the same static `let` initialization pattern. This means a model plugin does NOT need to register catalog components — only its own model-specific components. If both TuberiaCatalog and a model plugin register the same component ID, Acervo deduplicates silently (same ID + same repo = no-op).

SwiftTuberia addresses models exclusively through Acervo's abstractions:

- **Catalog queries**: `Acervo.isComponentReady(id)`, `Acervo.registeredComponents()`
- **Downloads**: `Acervo.ensureComponentsReady(recipe.allComponentIds)`
- **Weight access**: `AcervoManager.shared.withComponentAccess(id) { handle in ... }`

Pipeline code never constructs file paths, HuggingFace URLs, or hardcoded repo strings. If Acervo changes its storage layout, caching strategy, or download source — no pipeline code changes.

---

## Weight Loader

Loads safetensors files into `ModuleParameters` (see [PROTOCOLS.md](PROTOCOLS.md)) with key remapping, tensor transforms, and quantization. The single centralized loading path — no pipe segment ever parses safetensors or accesses files directly.

**Loading pipeline** (internal to WeightLoader):
```
1. withComponentAccess(componentId) → ComponentHandle
2. handle.urls(matching: ".safetensors") → file URLs (handles sharded weights)
3. For each safetensors file, for each key:
   a. keyMapping(originalKey) → remappedKey (nil = skip)
   b. tensorTransform?(remappedKey, tensor) → transformed tensor (nil = identity)
   c. Apply quantization per QuantizationConfig
4. Collect all key-tensor pairs → ModuleParameters
```

The WeightLoader obtains file access through `AcervoManager.shared.withComponentAccess()`. It never receives or stores file paths directly. URLs are valid only within the `withComponentAccess` closure scope.

**Capabilities**:
- Safetensors parsing (single file and sharded)
- Key remapping via `KeyMapping` closure (provided by `WeightedSegment.keyMapping`)
- Per-tensor transforms via `TensorTransform` (e.g., Conv2d weight transposition, provided by `WeightedSegment.tensorTransform`)
- Dtype conversion (float32 → float16/bfloat16)
- Post-load quantization (int4, int8 with configurable group size) via `QuantizationConfig`
- Progressive loading (stream keys to reduce peak memory)

Model plugins provide their key mapping and optional tensor transform via `WeightedSegment` conformance. The WeightLoader does all the mechanical work. Pipe segments receive clean `ModuleParameters` through `apply(weights:)` and never touch file I/O.

---

## Memory Manager

Coordinates memory across all loaded pipe segments. Global singleton actor.

**Available memory** uses Mach VM statistics including reclaimable pages (free + inactive + purgeable + speculative), giving a realistic picture of usable memory rather than just "free" pages.

**Cross-pipeline coordination**: MemoryManager tracks all loaded components across all pipelines (image, TTS, etc.). It reports total loaded memory but does NOT auto-unload — the caller/app decides priority. If budget is tight, MemoryManager returns `false`/throws and the caller decides what to evict.

**Headroom multiplier**: Per-consumer, not in MemoryManager. Each consumer applies its own headroom (e.g., VoxAlta 1.5× for KV caches, image pipelines 1.2×) and passes the multiplied value to `softCheck`/`hardValidate`. MemoryManager provides raw memory and loaded-component tracking.

**Single-gate coordination (REQ-PIPE-02, S4)**: `DiffusionPipeline.loadModels` calls `hardValidate(requiredBytes: peakMemoryBytes)` at entry via the `memoryGate` seam. If the budget is insufficient the call throws `PipelineError.insufficientMemory` before any weights are loaded. Phased loading with `softCheck` per phase is deferred to a future sortie — real peak-vs-phase divergence has not been observed. MemoryManager tracks what's loaded via `registerLoaded`/`unregisterLoaded` but does not decide the phase strategy.

---

## Device Capability Detection

Detection uses `sysctlbyname("machdep.cpu.brand_string")`. Neural Accelerator detection contributed from SwiftVoxAlta's M5 logic. Cached at first access.

**Access patterns**: `DeviceCapability.current` is the **synchronous** static property — it reads hardware info once and caches. This is the recommended accessor for all consumers. `MemoryManager.shared.deviceCapability` provides the same value through the actor for contexts that already have an actor reference, but requires `await`. For synchronous contexts (e.g., `EngineRouter` deciding which engines to register), use `DeviceCapability.current` directly — no actor hop needed.

---

## Canonical Swift Definitions

These are the **authoritative** infrastructure type definitions. If any prose above differs from the code below, **this code governs**.

All types below are `public` and live in the `Tuberia` target.

```swift
// MARK: - Weight Loader

public struct WeightLoader {
    public static func load(
        componentId: String,
        keyMapping: KeyMapping,
        tensorTransform: TensorTransform? = nil,
        quantization: QuantizationConfig = .asStored
    ) async throws -> ModuleParameters
}

// MARK: - Memory Manager

public actor MemoryManager {
    public static let shared: MemoryManager

    public var availableMemory: UInt64 { get }
    public var totalMemory: UInt64 { get }
    public var deviceCapability: DeviceCapability { get }

    public func softCheck(requiredBytes: UInt64) -> Bool
    public func hardValidate(requiredBytes: UInt64) throws

    public func registerLoaded(component: String, bytes: UInt64)
    public func unregisterLoaded(component: String)
    public var loadedComponentsMemory: UInt64 { get }

    public func clearGPUCache()
}

// MARK: - Device Capability

public struct DeviceCapability: Sendable {
    public let chipGeneration: AppleSiliconGeneration
    public let totalMemoryGB: Int
    public let platform: Platform
    public let hasNeuralAccelerators: Bool

    /// Synchronous static accessor. Cached at first access. No actor required.
    /// This is the recommended accessor for all consumers.
    public static let current: DeviceCapability

    public enum AppleSiliconGeneration: String, Sendable, CaseIterable {
        case m1, m1Pro, m1Max, m1Ultra
        case m2, m2Pro, m2Max, m2Ultra
        case m3, m3Pro, m3Max, m3Ultra
        case m4, m4Pro, m4Max, m4Ultra
        case m5, m5Pro, m5Max, m5Ultra
        case unknown
    }

    public enum Platform: String, Sendable {
        case macOS, iPadOS
    }
}
```

---

## SwiftAcervo Integration Compliance

**Status (2026-04-20)**: 🟡 PARTIAL — Scoped v2 access path is in place; integrity
verification, pre-load download, and memory budgeting are still outstanding. See
`/Users/stovak/Projects/SwiftTuberia/REQUIREMENTS.md` for the tracked sortie list and
`AUDIT_FINDINGS.md` for the original T1–T5 audit.

### What Is Implemented Today

- ✅ **No duplicate registry type** — `TuberiaCatalog` imports and uses
  `SwiftAcervo.ComponentDescriptor` directly
  (`Sources/TuberiaCatalog/Registration/CatalogRegistration.swift`).
- ✅ **`Acervo.register([…])` pattern** — module-level `let` trigger registers the
  catalog's descriptors on first import
  (`CatalogRegistration.swift:85-90`, via `CatalogRegistration.shared.ensureRegistered()`).
- ✅ **`withComponentAccess` for weights** — `WeightLoader.load(componentId:…)` routes
  every safetensors read through `AcervoManager.shared.withComponentAccess(_:perform:)`
  (`Sources/Tuberia/Infrastructure/WeightLoader.swift:40`). File URLs are resolved via
  `handle.urls(matching: ".safetensors")` and are valid only inside the closure.
- ✅ **`withLocalAccess` for LoRA adapters** — `WeightLoader.loadFromPath` (used by
  `LoRALoader`) routes caller-supplied paths through
  `AcervoManager.shared.withLocalAccess(_:perform:)`
  (`WeightLoader.swift:132`). Tuberia no longer touches `FileManager` directly for LoRA.
- ✅ **Tokenizer resolution via v2 API** —
  `T5XXLEncoder.loadTokenizer()` uses `withComponentAccess` to resolve `tokenizer.json`
  from the same component that owns the weights
  (`Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift:62`). Fallback is placeholder
  tokenization, never direct path I/O.
- ✅ **Error remapping** — `AcervoError` cases surface as
  `PipelineError.modelNotDownloaded` or `PipelineError.weightLoadingFailed`
  (`WeightLoader.swift:95-109`).
- ✅ **CatalogRegistration helper surface** — `CatalogRegistration.shared` exposes
  `ensureRegistered()`, `descriptor(for:)`, `isComponentRegistered(_:)`, and an async
  `ensureComponentReady(_:)` that delegates to `Acervo.ensureComponentReady`
  (`CatalogRegistration.swift:118-140`). The pipeline does not yet call the latter
  (see below).

### What Is NOT Yet Implemented

These bullets replace the previous "all complete" audit checklist; every item has a
corresponding sortie in `REQUIREMENTS.md`.

- ❌ **SHA-256 checksums** — every `ComponentFile` in `CatalogRegistration.swift`
  omits `sha256:` and `expectedSizeBytes:`. The integrity-verification loop inside
  `AcervoManager.withComponentAccess` (which checks `guard let expectedHash = file.sha256`)
  short-circuits for every file. (REQ-T4)
- ❌ **`Acervo.ensureComponentReady` in load path** — `DiffusionPipeline.loadModels`
  does not pre-download. A first-run call with an empty cache throws
  `AcervoError.componentNotDownloaded` from inside `withComponentAccess`. The
  "loading flow" block above (step 1) is the **target** behavior, not current code.
  (REQ-PIPE-01)
- ✅ **`MemoryManager.hardValidate()` gate in load path** — `DiffusionPipeline.loadModels`
  now calls `MemoryManager.shared.hardValidate(requiredBytes: peakMemoryBytes)` at entry
  via the `memoryGate` seam (single up-front gate; phased `softCheck` is deferred).
  `PipelineError.insufficientMemory(required:available:component:)` surfaces when the
  budget is exceeded. `MemoryGuardTests.swift` covers the failure and pass-through paths.
  (REQ-PIPE-02)
- ❌ **Positional component-id lookup** — `DiffusionPipeline.findComponentId(for:)`
  uses `_allComponentIds[0|1|2]` as encoder/backbone/decoder. Any recipe that declares
  IDs in a different order mis-associates weights. (REQ-PIPE-03)
- ❌ **Dependency floor** — `Package.swift` still declares `SwiftAcervo` at
  `from: "0.5.6"` even though the resolved version is 0.7.2 and v2 symbols only ship
  from 0.7.x. Fresh resolutions can regress. (REQ-T5)
- ❌ **v2 integration tests** — no test in `Tests/` invokes `withComponentAccess`,
  `WeightLoader.load`, or forces an `integrityCheckFailed`. (REQ-INT-01)

### Pattern Details (As Implemented)

```
DiffusionPipeline.loadModels(progress:)
  ↓
  try await memoryGate(peakMemoryBytes)         // hardValidate gate (REQ-PIPE-02, S4)
    → throws PipelineError.insufficientMemory   // if available < peakMemoryBytes
  ↓
for segment in [encoder, backbone, decoder]:
  componentId = _componentIdByRole[role]        // role-keyed dict (REQ-PIPE-03, S5)
  try await ensureComponentReady(componentId)   // downloads if missing (REQ-PIPE-01, S3)
  weights = try await WeightLoader.load(
      componentId: componentId,
      keyMapping: segment.keyMapping,
      tensorTransform: segment.tensorTransform,
      quantization: recipe.quantizationFor(role)
  )
  ↓
  WeightLoader.load routes through:
    AcervoManager.shared.withComponentAccess(componentId) { handle in
        let urls = try handle.urls(matching: ".safetensors")
        // parse, remap, transform, quantize — paths valid only in this closure
        return ModuleParameters(parameters: ...)
    }
  ↓
  segment.apply(weights: weights)
  await MemoryManager.shared.registerLoaded(component: componentId, bytes: …)
```

**Key properties holding today**:
- No paths escape the `withComponentAccess` / `withLocalAccess` closure.
- Sharded weights are supported (the handle returns all matching `.safetensors` URLs).
- LoRA reuse the same closure-scoped discipline via `withLocalAccess`.
- Single up-front memory gate (`hardValidate`) fires before any I/O.

**Key properties still missing** (see sortie list above):
- Phased `softCheck` per loading phase (deferred; single up-front gate is in place).
- No integrity verification in practice (all `sha256` are `nil`).
- No memory budget gate before allocating.

### MACF / Sandboxed Enumeration Behavior

The previous revision of this document described a "MACF workaround via
`VINETAS_TEST_MODELS_DIR`". That environment variable has been removed from the
codebase; `grep VINETAS_TEST_MODELS_DIR Sources/` returns no hits. The current
behavior is:

- `WeightLoader.findSafetensorsFiles(in:)` tries `FileManager.enumerator` first.
- If enumeration returns zero entries (which includes the case where the process can
  stat but not `opendir` an App Group Container directory), it falls through to a
  stat-based probe that covers the two HuggingFace naming conventions
  (`model.safetensors` and `model-NNNNN-of-MMMMM.safetensors`).
- This allows unentitled xctest processes to still locate files that `opendir` is
  denied for.

The private `canEnumerateDirectory(_:)` helper
(`WeightLoader.swift:168`) is declared but no longer called — it is dead code left
over from v0.3.2/0.3.5 refactors. Consider pruning it in a cleanup sortie.

### Component Registration (As Implemented)

```swift
// Sources/TuberiaCatalog/Registration/CatalogRegistration.swift

private let t5XXLEncoderComponentDescriptor = SwiftAcervo.ComponentDescriptor(
    id: "t5-xxl-encoder-int4",
    type: .encoder,
    displayName: "T5-XXL Text Encoder (int4)",
    repoId: "intrusive-memory/t5-xxl-int4-mlx",
    files: t5XXLEncoderRequiredFiles,            // sha256: nil — see REQ-T4
    estimatedSizeBytes: 1_288_490_188,
    minimumMemoryBytes: 2_000_000_000,
    metadata: [...]
)

private let sdxlVAEDecoderComponentDescriptor = SwiftAcervo.ComponentDescriptor(
    id: "sdxl-vae-decoder-fp16",
    type: .decoder,
    ...
)

private let _registerTuberiaCatalogComponents: Void = {
    Acervo.register([
        t5XXLEncoderComponentDescriptor,
        sdxlVAEDecoderComponentDescriptor
    ])
}()
```

Consumers that need to force registration before any access (e.g. early in app launch)
can call `CatalogRegistration.shared.ensureRegistered()`. Importing `TuberiaCatalog`
also triggers the `let` initializer on first use.

### Testing State

- **Unit tests present**: `Tests/TuberiaCatalogTests/CatalogRegistrationTests.swift`
  validates that registration occurs, IDs are correct, metadata round-trips, and
  `files.count` matches the declared set. No integrity or access-path coverage.
- **No v2 integration coverage**: `grep withComponentAccess Tests/` → 0 hits.
- **MACF / env-var gating**: removed. Integration tests that previously guarded on
  `VINETAS_TEST_MODELS_DIR` have been rewritten to use synthetic `MLXArray`s; no
  weight-backed tests remain in the regular suites. See
  `requirements/TESTING.md` + `requirements/TEST_COVERAGE_GAPS.md` for what landed.

### Audit Checklist (Honest Boxes)

- [x] `AcervoManager.withComponentAccess()` used for scoped weight file access.
- [x] `AcervoManager.withLocalAccess()` used for LoRA adapters.
- [x] No file paths stored outside closure scope.
- [x] `TuberiaCatalog` registers `ComponentDescriptor`s via `Acervo.register(_:)` at
      module init.
- [x] Error mapping: `AcervoError` → `PipelineError.modelNotDownloaded` /
      `PipelineError.weightLoadingFailed`.
- [x] AGENTS.md documents component registration (section "Project Overview" + v0.3.0
      changelog entry).
- [ ] **SHA-256 checksums** populated on every `ComponentFile` (REQ-T4).
- [ ] **`Acervo.ensureComponentReady`** invoked before `WeightLoader.load`
      (REQ-PIPE-01).
- [x] **`MemoryManager.hardValidate()` invoked from the load path** — single up-front
      gate against `peakMemoryBytes` at entry of `loadModels(progress:)`; throws
      `PipelineError.insufficientMemory(required:available:component:)` on budget
      exhaustion. Phased `softCheck` is deferred (REQ-PIPE-02, S4).
      `MemoryGuardTests.swift` verifies the failure and pass-through paths.
- [ ] **Role-based component-id mapping** replaces positional
      `findComponentId(for:)` (REQ-PIPE-03).
- [ ] **`SwiftAcervo` dependency floor** bumped to ≥ 0.7.2 in `Package.swift`
      (REQ-T5).
- [ ] **End-to-end v2 integration tests**: happy-path, integrity failure,
      not-downloaded, LoRA local-access (REQ-INT-01).
- [ ] MACF workaround doc updated to reflect the current stat-based probe (done in
      this document; remove stale `VINETAS_TEST_MODELS_DIR` references from AGENTS.md
      if present).

### Reference

- `/Users/stovak/Projects/SwiftTuberia/REQUIREMENTS.md` — tracked sortie list
  (REQ-T4, REQ-T5, REQ-PIPE-01/02/03, REQ-INT-01, REQ-CDN-01, REQ-DOC-01).
- `/Users/stovak/Projects/SwiftTuberia/AUDIT_FINDINGS.md` — original T1–T5 audit.
- `/Users/stovak/Projects/SwiftAcervo/API_REFERENCE.md` — upstream v2 API surface
  (`AcervoManager.withComponentAccess`, `ComponentFile.sha256`,
  `AcervoError.integrityCheckFailed`, etc.).
- `/Users/stovak/Projects/REQUIREMENTS.md` — master mission index (Work Unit 2).
