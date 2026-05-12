---
title: "SwiftTuberia — SwiftAcervo V2 Compliance Requirements"
date: 2026-04-27
status: "OPEN"
audit_basis: "SwiftAcervo 0.8.2 API surface audit"
v2_goal: "SwiftTuberia never deals with files. Acervo provides all model access."
---

# SwiftAcervo V2 Compliance Requirements

**V2 design contract**: SwiftTuberia must never construct file paths, enumerate
directories, compute checksums, or know file names for any component managed by
Acervo. All file access flows through `ComponentHandle` or `LocalHandle` APIs.
Acervo is the sole authority on what files exist, where they live, and whether
they are intact.

---

## Audit Findings

### What is already correct

- `WeightLoader.load()` uses `handle.urls(matching: ".safetensors")` exclusively — no filesystem in the happy path.
- `WeightLoader.loadFromPath()` routes local LoRA paths through `AcervoManager.withLocalAccess`.
- `T5XXLEncoder.loadTokenizer()` uses `AcervoManager.withComponentAccess` with `handle.url(matching:)`.
- `ComponentReadinessService` calls `Acervo.ensureComponentReady` with a progress callback before any load.
- `DiffusionPipeline.loadModels()` gates on `ensureComponentReady` before calling `WeightLoader`.
- No `URLSession` downloads, no `shasum` computation, no manual caching logic anywhere in SwiftTuberia.

### What is broken or incomplete

See requirements below. Priority order: correctness gaps first, then abstraction
gaps, then cleanliness.

---

## Requirements

### TUBERIA-V2-01 — Adopt bare descriptor pattern in CatalogRegistration.swift [CRITICAL]

**Files**: `Sources/TuberiaCatalog/Registration/CatalogRegistration.swift:14–81`

**Problem**: All `ComponentFile` entries have `sha256: nil` and `expectedSizeBytes: nil`.
`AcervoManager.withComponentAccess` (AcervoManager.swift:481) skips per-file
integrity verification when `sha256 == nil`:

```swift
for file in descriptor.files {
    guard let expectedHash = file.sha256 else { continue }  // ← skips nil checksums
    // ... SHA-256 check never runs
}
```

This means every `withComponentAccess` call for T5-XXL and SDXL VAE performs only a
file-existence check, not an integrity check. A corrupted or tampered weight file
passes through silently.

With bare descriptors, `ensureComponentReady` fetches the CDN manifest and replaces
the file list with manifest entries that carry real `sha256` and `expectedSizeBytes`
values. After hydration, every `withComponentAccess` call runs the per-file SHA-256
check.

Additionally, the hardcoded 11 file names (9 T5 shards + 2 VAE files) couple
SwiftTuberia to the CDN repo file structure. Any shard count change, file rename, or
new file requires source code changes here.

**Required change**: Replace both full-initializer descriptors with bare descriptors.

```swift
// Before
private let t5XXLEncoderRequiredFiles: [ComponentFile] = [
    ComponentFile(relativePath: "config.json", expectedSizeBytes: nil, sha256: nil),
    // ... 8 more hardcoded files
]

private let t5XXLEncoderComponentDescriptor = ComponentDescriptor(
    id: "t5-xxl-encoder-int4",
    type: .encoder,
    displayName: "T5-XXL Text Encoder (int4)",
    repoId: "intrusive-memory/t5-xxl-int4-mlx",
    files: t5XXLEncoderRequiredFiles,          // ← hardcoded, nil checksums
    estimatedSizeBytes: 1_288_490_188,
    minimumMemoryBytes: 2_000_000_000,
    metadata: [...]
)

// After
private let t5XXLEncoderComponentDescriptor = ComponentDescriptor(
    id: "t5-xxl-encoder-int4",
    type: .encoder,
    displayName: "T5-XXL Text Encoder (int4)",
    repoId: "intrusive-memory/t5-xxl-int4-mlx",
    minimumMemoryBytes: 2_000_000_000,        // ← bare: CDN manifest provides files
    metadata: [...]
)
```

Delete `t5XXLEncoderRequiredFiles` and `sdxlVAEDecoderRequiredFiles` entirely.

**Trade-off**: Before first hydration, `estimatedSizeBytes == 0` — the catalog size UI
will show 0 until `ensureComponentReady` runs. This is acceptable: the sizes come
from the manifest, which is the authoritative source. Do not add a hardcoded size
constant as a compensating hack; accept the 0 pre-hydration.

**Acceptance criteria**:
- `grep -c 'ComponentFile(' Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` returns `0`
- `grep -c 'sha256:' Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` returns `0`
- `grep -c 'files:' Sources/TuberiaCatalog/Registration/CatalogRegistration.swift` returns `0`
- `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO` exits 0

**Dependency**: ACERVO-V2-01 (bare descriptor hydration is already implemented in
0.8.2; this requirement depends only on the existing bare initializer).

---

### TUBERIA-V2-02 — Remove dead filesystem methods from WeightLoader.swift [CRITICAL]

**Files**: `Sources/Tuberia/Infrastructure/WeightLoader.swift:163–239`

**Problem**: Two private methods remain in `WeightLoader` that perform direct
filesystem operations:

- `canEnumerateDirectory(_:)` (lines 168–177): calls `FileManager.default.enumerator`
- `findSafetensorsFiles(in:)` (lines 187–239): calls `FileManager.enumerator` with a
  stat-based probe fallback using `fm.fileExists` and manual shard name construction

Neither method is called anywhere. They are dead code. Their presence:

1. Contradicts the file header assertion ("no pipe segment ever parses safetensors or
   accesses files directly")
2. Retains knowledge of HuggingFace shard naming conventions (`model-NNNNN-of-MMMMM`)
   and single-file conventions (`model.safetensors`) that belong in Acervo, not here
3. Created a prior MACF (Mandatory Access Control Framework) bypass workaround for
   App Group containers that should have been resolved in Acervo's `ComponentHandle`

The MACF concern (App Group containers where `opendir` is denied but `stat` works)
belongs inside `ComponentHandle.urls(matching:)` in SwiftAcervo, not in every
consumer. If it resurfaces, the fix is there.

**Required change**: Delete both methods entirely.

```swift
// Delete lines 163–239:
//   canEnumerateDirectory(_:)
//   findSafetensorsFiles(in:)
```

No callers to update.

**Acceptance criteria**:
- `grep -n 'findSafetensorsFiles\|canEnumerateDirectory\|FileManager' Sources/Tuberia/Infrastructure/WeightLoader.swift` returns 0 matches
- `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO` exits 0

---

### TUBERIA-V2-03 — Replace string-matching AcervoError detection in WeightLoader.swift [IMPORTANT]

**Files**: `Sources/Tuberia/Infrastructure/WeightLoader.swift:98–108`

**Problem**: The catch block identifies Acervo errors by searching the error's string
description:

```swift
} catch {
    let errorString = String(describing: error)
    if errorString.contains("NotDownloaded") || errorString.contains("notDownloaded")
        || errorString.contains("notRegistered") || errorString.contains("invalidModelId")
    {
        throw PipelineError.modelNotDownloaded(component: componentId)
    }
    throw PipelineError.weightLoadingFailed(component: componentId, reason: errorString)
}
```

This breaks if `AcervoError` case names or descriptions change, and it misses any
Acervo error case not in the hardcoded list. `AcervoError` is a `public enum` with
named cases — pattern matching is the correct and stable approach.

**Required change**: Replace string inspection with typed pattern matching.

```swift
} catch AcervoError.componentNotDownloaded {
    throw PipelineError.modelNotDownloaded(component: componentId)
} catch AcervoError.componentNotRegistered {
    throw PipelineError.modelNotDownloaded(component: componentId)
} catch AcervoError.componentNotHydrated {
    throw PipelineError.modelNotDownloaded(component: componentId)
} catch AcervoError.integrityCheckFailed(let file, let expected, let actual) {
    throw PipelineError.weightLoadingFailed(
        component: componentId,
        reason: "Integrity check failed for \(file): expected \(expected), got \(actual)"
    )
} catch let error as PipelineError {
    throw error
} catch {
    throw PipelineError.weightLoadingFailed(
        component: componentId,
        reason: String(describing: error)
    )
}
```

Apply the same pattern to `loadFromPath`'s catch block (lines 151–158).

**Acceptance criteria**:
- `grep -n 'contains("Not\|contains("not\|contains("invalid' Sources/Tuberia/Infrastructure/WeightLoader.swift` returns 0 matches
- `grep -n 'AcervoError\.' Sources/Tuberia/Infrastructure/WeightLoader.swift` returns ≥ 3 matches (one per caught case)

---

### ACERVO-V2-01 — Add `public var rootDirectoryURL: URL` to ComponentHandle [IMPORTANT]

**Files**: `../SwiftAcervo/Sources/SwiftAcervo/ComponentHandle.swift`

**Problem**: `ComponentHandle.baseDirectory` is `internal`. Callers that need the
component's root directory — specifically for APIs that require a directory URL rather
than a single file URL — must derive it indirectly:

```swift
// Current workaround in T5XXLEncoder.loadTokenizer():
let tokenizerURL = try handle.url(matching: "tokenizer.json")
return tokenizerURL.deletingLastPathComponent()   // ← path manipulation
```

This is fragile: it assumes `tokenizer.json` is in the component root. It would break
if Acervo ever uses a staging directory, version subdirectory, or any other layout
that puts a level of indirection between `baseDirectory` and the file.

`LocalHandle` already exposes `public let rootURL: URL`. `ComponentHandle` should
match that API.

**Required change in SwiftAcervo**: Expose `baseDirectory` as a public property.

```swift
// ComponentHandle.swift — add one line
public var rootDirectoryURL: URL { baseDirectory }
```

No behavior change; pure API surface addition.

**Acceptance criteria (in SwiftAcervo)**:
- `grep -n 'public var rootDirectoryURL' Sources/SwiftAcervo/ComponentHandle.swift` returns 1 match
- Existing `ComponentHandle` tests pass
- `ComponentHandle` and `LocalHandle` now have matching "root URL" accessor names

---

### TUBERIA-V2-04 — Use `handle.rootDirectoryURL` in T5XXLEncoder.loadTokenizer [IMPORTANT]

**Files**: `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift:67–72`

**Blocked by**: ACERVO-V2-01

**Problem**: `deletingLastPathComponent()` assumes the tokenizer files are direct
children of the component directory. If the Acervo layout ever adds indirection, this
silently loads the wrong directory.

**Required change** (after ACERVO-V2-01 ships in SwiftAcervo):

```swift
// Before
let tokenizerURL = try handle.url(matching: "tokenizer.json")
return tokenizerURL.deletingLastPathComponent()

// After
return handle.rootDirectoryURL
```

The `withComponentAccess` closure then returns `handle.rootDirectoryURL` directly,
and `AutoTokenizer.from(directory: tokenizerDir)` receives the canonical component
root.

**Acceptance criteria**:
- `grep -n 'deletingLastPathComponent' Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift` returns 0 matches
- Tokenizer loads correctly in TuberiaCatalogTests

---

### TUBERIA-V2-05 — Document ensureComponentReady precondition in T5XXLEncoder.loadTokenizer [MINOR]

**Files**: `Sources/TuberiaCatalog/Encoders/T5XXLEncoder.swift:58–82`

**Problem**: `loadTokenizer()` calls `withComponentAccess` without calling
`ensureComponentReady` first. It relies on the pipeline having already done so in
`DiffusionPipeline.loadModels()`. This implicit ordering dependency is not documented.

With bare descriptors (after TUBERIA-V2-01), `withComponentAccess` will fail if the
component is not hydrated — so the ordering dependency becomes a hard correctness
requirement, not just a nice-to-have.

**Required change**: Add a precondition note to the `loadTokenizer` docstring:

```swift
/// Precondition: The component identified by `configuration.componentId` must have
/// been prepared via `Acervo.ensureComponentReady` (or equivalent) before this
/// method is called. `DiffusionPipeline.loadModels()` satisfies this precondition.
/// Calling this method before the component is ready will throw `componentNotHydrated`
/// or `componentNotDownloaded`.
```

**Acceptance criteria**:
- Docstring contains "ensureComponentReady" and "precondition" (or equivalent)

---

### TUBERIA-V2-06 — Deprecate CatalogRegistration.ensureComponentReady [MINOR]

**Files**: `Sources/TuberiaCatalog/Registration/CatalogRegistration.swift:139–142`

**Problem**: `CatalogRegistration.shared.ensureComponentReady(_:)` is a thin wrapper
that omits the progress callback:

```swift
public func ensureComponentReady(_ componentId: String) async throws {
    try await Acervo.ensureComponentReady(componentId)
}
```

The canonical pipeline path is `ComponentReadinessService.ensureComponentReady(_:progress:)`.
The wrapper creates a second entry point with weaker capability (no progress) and no
clear advantage over calling `Acervo.ensureComponentReady` directly. Callers should
use the protocol-backed `ComponentReadinessService` seam instead.

**Required change**: Mark as deprecated with a migration note.

```swift
@available(*, deprecated, message: "Use ComponentReadinessService (with progress) or Acervo.ensureComponentReady directly.")
public func ensureComponentReady(_ componentId: String) async throws {
    try await Acervo.ensureComponentReady(componentId)
}
```

Do not delete yet — verify no callers first, then remove in a follow-up.

**Acceptance criteria**:
- `@available(*, deprecated` annotation present on the method

---

### ACERVO-V2-02 — Allow estimated size hint on bare ComponentDescriptor [OPTIONAL]

**Files**: `../SwiftAcervo/Sources/SwiftAcervo/ComponentDescriptor.swift:144–161`

**Problem**: The bare initializer does not accept `estimatedSizeBytes`. Pre-hydration,
`descriptor.estimatedSizeBytes == 0`, making `Acervo.totalCatalogSize()` return 0 for
un-hydrated components. Download size estimates in UI are unavailable until the first
`ensureComponentReady` call per app launch.

**Required change in SwiftAcervo** (optional, quality-of-life):

```swift
public init(
    id: String,
    type: ComponentType,
    displayName: String,
    repoId: String,
    minimumMemoryBytes: Int64,
    estimatedSizeBytes: Int64? = nil,  // ← new optional hint
    metadata: [String: String] = [:]
) {
    ...
    self._estimatedSizeBytes = estimatedSizeBytes
    ...
}
```

If provided, this is a documentation-only hint for pre-hydration UI. The manifest's
sum of `sizeBytes` replaces it after hydration.

**Acceptance criteria (in SwiftAcervo)**:
- `ComponentDescriptor` bare initializer accepts `estimatedSizeBytes: Int64? = nil`
- `estimatedSizeBytes` property returns the hint before hydration, manifest sum after

---

## Implementation Order

| # | Requirement | Target | Blocks |
|---|-------------|--------|--------|
| 1 | TUBERIA-V2-01 | SwiftTuberia | nothing (correctness gap) |
| 2 | TUBERIA-V2-02 | SwiftTuberia | nothing (dead code) |
| 3 | TUBERIA-V2-03 | SwiftTuberia | nothing (error handling) |
| 4 | ACERVO-V2-01 | SwiftAcervo | TUBERIA-V2-04 |
| 5 | TUBERIA-V2-04 | SwiftTuberia | requires ACERVO-V2-01 |
| 6 | TUBERIA-V2-05 | SwiftTuberia | none (docs) |
| 7 | TUBERIA-V2-06 | SwiftTuberia | none (deprecation) |
| 8 | ACERVO-V2-02 | SwiftAcervo | none (optional) |

Requirements 1–3 are independent of each other and can be implemented in parallel.
Requirement 4 must ship in SwiftAcervo before Requirement 5 can land in SwiftTuberia.
Requirements 6–7 are pure annotations, no behavior change.

---

## Definition of Done

The V2 contract is satisfied when:

1. `grep -rn 'ComponentFile\|findSafetensorsFiles\|canEnumerateDirectory\|FileManager' Sources/` returns 0 matches
2. `grep -rn 'contains("Not\|contains("not' Sources/Tuberia/Infrastructure/WeightLoader.swift` returns 0 matches
3. `grep -rn 'deletingLastPathComponent' Sources/TuberiaCatalog/` returns 0 matches
4. `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO` exits 0
5. All `ComponentHandle` and `LocalHandle` accesses in SwiftTuberia go through `.url(for:)`, `.url(matching:)`, `.urls(matching:)`, or `.rootDirectoryURL` — no path construction, no `FileManager`, no directory enumeration

At that point, SwiftTuberia has zero knowledge of model file names, zero knowledge of
storage locations, and zero responsibility for integrity verification.
