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

**Two-phase coordination**: Pipeline initiates, MemoryManager coordinates. Pipeline queries `softCheck(peakMemoryBytes)`. If insufficient, pipeline falls back to phased loading, calling `clearGPUCache()` between phases. MemoryManager tracks what's loaded via `registerLoaded`/`unregisterLoaded` but does not decide the phase strategy.

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
