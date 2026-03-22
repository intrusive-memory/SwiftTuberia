# SwiftTubería — Infrastructure Architecture

**Companion to**: [`../requirements/INFRASTRUCTURE.md`](../requirements/INFRASTRUCTURE.md)
**Target**: `Tubería`

---

## Infrastructure Services

| Service | Type | Access Pattern | Consumed By |
|---|---|---|---|
| `WeightLoader` | Struct (static method) | `WeightLoader.load(...)` | DiffusionPipeline (internal) |
| `MemoryManager` | Actor (singleton) | `MemoryManager.shared` | All pipelines, SwiftVoxAlta |
| `DeviceCapability` | Struct (cached static) | `DeviceCapability.current` | SwiftVinetas (engine registration), SwiftVoxAlta |

---

## WeightLoader

```swift
struct WeightLoader {
    static func load(
        componentId: String,                    // Acervo component ID
        keyMapping: KeyMapping,                 // From WeightedSegment
        tensorTransform: TensorTransform? = nil,// From WeightedSegment (optional)
        quantization: QuantizationConfig = .asStored // From PipelineRecipe
    ) async throws -> ModuleParameters
}
```

### Internal Loading Pipeline

```
1. AcervoManager.shared.withComponentAccess(componentId) { handle in
2.   handle.urls(matching: ".safetensors") → [URL]  (handles shards)
3.   For each file, for each key:
       a. keyMapping(originalKey) → remappedKey (nil = skip)
       b. tensorTransform?(remappedKey, tensor) → transformed tensor
       c. Apply QuantizationConfig
4.   Return ModuleParameters(parameters: [String: MLXArray])
   }
```

**Key principle**: WeightLoader is the ONLY code that reads safetensors. Pipe segments never touch file I/O.

### Data Flow

```
Acervo ComponentHandle (URLs)
    → safetensors parsing
    → KeyMapping (from WeightedSegment)
    → TensorTransform (from WeightedSegment, optional)
    → QuantizationConfig (from PipelineRecipe)
    → ModuleParameters
    → segment.apply(weights:)
```

---

## MemoryManager

```swift
actor MemoryManager {
    static let shared: MemoryManager

    // Query
    var availableMemory: UInt64 { get }         // Mach VM: free+inactive+purgeable+speculative
    var totalMemory: UInt64 { get }
    var deviceCapability: DeviceCapability { get }

    // Validation
    func softCheck(requiredBytes: UInt64) -> Bool
    func hardValidate(requiredBytes: UInt64) throws    // throws PipelineError.insufficientMemory

    // Tracking
    func registerLoaded(component: String, bytes: UInt64)
    func unregisterLoaded(component: String)
    var loadedComponentsMemory: UInt64 { get }

    // GPU
    func clearGPUCache()                        // MLX GPU sync + clear cache + update tracking
}
```

### Consumer Usage Patterns

**DiffusionPipeline** (internal):
```
loadModels() → softCheck(peakMemoryBytes)
  if insufficient → fallback to phased loading
  between phases → clearGPUCache()
  after each segment load → registerLoaded(component, bytes)
```

**SwiftVoxAlta** (external):
```
validateMemory(for modelId):
  let requiredBytes = descriptor.minimumMemoryBytes * 1.5  // VoxAlta's headroom
  softCheck(requiredBytes) or hardValidate(requiredBytes)
```

**Headroom multiplier**: Per-consumer, NOT in MemoryManager. VoxAlta applies 1.5x, image pipelines apply 1.2x. MemoryManager provides raw memory data.

---

## DeviceCapability

```swift
struct DeviceCapability: Sendable {
    let chipGeneration: AppleSiliconGeneration   // .m1 through .m5Ultra, .unknown
    let totalMemoryGB: Int
    let platform: Platform                        // .macOS, .iPadOS
    let hasNeuralAccelerators: Bool

    static let current: DeviceCapability          // Synchronous, cached at first access
}
```

### Consumer Usage

**SwiftVinetas** (engine registration):
```swift
var engines: [any ImageGenerationEngine] = [PixArtEngine()]
if DeviceCapability.current.totalMemoryGB >= 16 {
    engines.append(Flux2Engine())
}
```

**SwiftVoxAlta** (replaces AppleSiliconInfo):
```swift
let device = DeviceCapability.current
// device.chipGeneration, device.hasNeuralAccelerators, device.totalMemoryGB
```

---

## Cross-Consumer Memory Coordination

```
                    MemoryManager.shared
                   ╱                    ╲
    DiffusionPipeline                SwiftVoxAlta
    (image model loaded)             (TTS model loaded)
         │                                │
         ├── registerLoaded               ├── registerLoaded
         │   ("pixart-dit", 300MB)        │   ("qwen3-tts-1.7b", 3.4GB)
         │                                │
         └── loadedComponentsMemory ──────┴── reports total: 3.7 GB

Both consumers tracked. Neither auto-evicts. App decides priority.
```
