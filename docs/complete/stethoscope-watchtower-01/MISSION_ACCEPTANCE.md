---
mission: OPERATION STETHOSCOPE WATCHTOWER
sortie: 13 of 13 (terminal)
date: 2026-05-07
branch: mission/stethoscope-watchtower/01
starting_point_commit: 37aeb415c9236983ad09296f8adced44be9df4a2
verdict: READY TO SHIP
---

# MISSION ACCEPTANCE — OPERATION STETHOSCOPE WATCHTOWER

All 12 prior sorties completed. This document records the Sortie 13 acceptance
verification against REQUIREMENTS-telemetry.md §12 and the decision on the
cross-actor telemetry-forwarding gap.

---

## Entry Criteria Verification

| Check | Command | Result |
|-------|---------|--------|
| All 12 sorties completed | `git log --oneline -15` | PASS — commits 0f356f3 through eeb40cb (13 commits including scaffolding) present |
| Working tree clean | `git status --porcelain` | PASS — only SUPERVISOR_STATE.md has pre-existing supervisor-managed changes; no uncommitted code changes |
| On mission branch | `git branch --show-current` | PASS — `mission/stethoscope-watchtower/01` |

---

## §12 Acceptance Checklist

| # | Item | Verifying Command | Result |
|---|------|-------------------|--------|
| 1 | `Sources/Tuberia/Telemetry/{TuberiaTelemetryEvent,TuberiaTelemetryReporter,NoopTuberiaTelemetryReporter}.swift` exist and are exported from the `Tuberia` product | `test -f Sources/Tuberia/Telemetry/TuberiaTelemetryEvent.swift && test -f Sources/Tuberia/Telemetry/TuberiaTelemetryReporter.swift && test -f Sources/Tuberia/Telemetry/NoopTuberiaTelemetryReporter.swift && echo OK` | PASS — all 3 files exist |
| 1a | Public declarations (enum, protocol, struct) present | `grep -nE "^public " Sources/Tuberia/Telemetry/*.swift` | PASS — `public enum TuberiaTelemetryEvent: Sendable` (TuberiaTelemetryEvent.swift:7), `public protocol TuberiaTelemetryReporter: Sendable` (TuberiaTelemetryReporter.swift:14), `public struct NoopTuberiaTelemetryReporter` (NoopTuberiaTelemetryReporter.swift:6) |
| 2 | `DiffusionPipeline.setTelemetry(_:)` exists | `grep -n "setTelemetry" Sources/Tuberia/Pipeline/DiffusionPipeline.swift` | PASS — line 203: `public func setTelemetry(_ reporter: (any TuberiaTelemetryReporter)?)` |
| 2a | `MemoryManager.setTelemetry(_:)` exists | `grep -n "setTelemetry" Sources/Tuberia/Infrastructure/MemoryManager.swift` | PASS — line 41: `public func setTelemetry(_ reporter: (any TuberiaTelemetryReporter)?)` |
| 2b | `WeightLoader.load(…, telemetry:)` defaulted parameter exists | `grep -n "telemetry: (any TuberiaTelemetryReporter)? = nil" Sources/Tuberia/Infrastructure/WeightLoader.swift` | PASS — found at lines 44 and 145 (load + loadFromPath) |
| 3 | All instrumentation points emit when reporter attached and emit nothing when it isn't | `xcodebuild test … -parallel-testing-enabled NO` (6 telemetry test suites, 81 total tests) | PASS — all 81 tests pass; DiffusionPipelineTelemetryTests, GenerateTelemetryTests, LoRAErrorTelemetryTests, MemoryManagerTelemetryTests, WeightLoaderTelemetryTests, TuberiaTelemetryEventTests all included |
| 4 | `xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO` passes including new Telemetry suite | Full suite run | PASS — **81 tests in 12 suites passed** in 0.128 seconds. `TelemetryOverheadTests` correctly absent (gated by `TUBERIA_TELEMETRY_OVERHEAD_ENABLED` compile flag) |
| 5 | No new dependencies in `Package.swift` | `git diff 37aeb415c9236983ad09296f8adced44be9df4a2 -- Package.swift` | PASS — zero diff output; Package.swift is byte-identical to starting point |
| 6 | `AGENTS.md` and `README.md` updated; consumer-facing snippet present | `grep -n "setTelemetry\|TuberiaTelemetryReporter" README.md` and `grep -n "## Telemetry" AGENTS.md` | PASS — README.md line 93 (`pipeline.setTelemetry`), line 108 (`TuberiaTelemetryReporter`); AGENTS.md line 248 (`## Telemetry`), line 289 (`### How Produciesta consumes this`) |
| 7 | Produciesta engineer can wire up a `TuberiaTelemetryAdapter` without Tuberia importing Produciesta | `grep -r "import Produciesta\|import ProduciestaCore" Sources/Tuberia/` | PASS — zero matches |

---

## Additional Verification Checks

| Check | Command | Result |
|-------|---------|--------|
| 6 telemetry test files present | `ls Tests/TuberiaTests/Telemetry/` | PASS — 8 files: MockTuberiaTelemetryReporter.swift, TuberiaTelemetryEventTests.swift, DiffusionPipelineTelemetryTests.swift, GenerateTelemetryTests.swift, LoRAErrorTelemetryTests.swift, WeightLoaderTelemetryTests.swift, MemoryManagerTelemetryTests.swift, TelemetryOverheadTests.swift |
| All test files use Swift Testing | `grep -rn "import Testing" Tests/TuberiaTests/Telemetry/*.swift` | PASS — all 8 files import Testing |
| `make test-telemetry-overhead` target exists | `grep -n "test-telemetry-overhead:" Makefile` | PASS — line 31 |
| Makefile overhead target uses `-parallel-testing-enabled NO` | `grep -A3 "test-telemetry-overhead:" Makefile \| grep -c "parallel-testing-enabled NO"` | PASS — count 1 |
| `make test` does NOT include TelemetryOverheadTests | Default `make test` invocation (no `TUBERIA_TELEMETRY_OVERHEAD_ENABLED`) | PASS — gated by `#if TUBERIA_TELEMETRY_OVERHEAD_ENABLED` compile flag; suite compiles as empty stub with no `@Test` methods |
| Exactly 1 `unloadSegment(role: nil)` emit site | `grep -rn "\.unloadSegment(role: nil" Sources/Tuberia/` | PASS — MemoryManager.swift:148 only |
| QuantizationConfig.CustomStringConvertible extension | `grep -n "extension QuantizationConfig: CustomStringConvertible" Sources/Tuberia/Types/QuantizationConfig.swift` | PASS — line 10 |
| No `Produciesta` / `ProduciestaCore` imports in Sources | `grep -rn "import Produciesta" Sources/` | PASS — zero matches |

---

## Full Test Run Summary

**Command:**
```
xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO
```

**Result: TEST SUCCEEDED**

**Total: 81 tests in 12 suites passed in 0.128 seconds**

Suites (12 total):
- TuberiaTelemetryEvent Tests (25 tests)
- DiffusionPipeline Telemetry Tests (8 tests)
- Generate Telemetry Tests (3 tests)
- LoRA / errorThrown Telemetry Tests (5 tests)
- WeightLoader Telemetry Tests (3 tests)
- MemoryManager Telemetry Tests (5 tests)
- MemoryGuard Tests (3 tests)
- MemoryManager Tests (7 tests)
- RecipeRoleMap Tests (3 tests)
- LoRA Integration Tests (4 tests)
- WeightLoader Integration Tests (included in suite count)
- (additional pre-telemetry suites)

---

## Package.swift Dependency Diff

```
git diff 37aeb415c9236983ad09296f8adced44be9df4a2 -- Package.swift
```

Output: (empty — zero diff)

**Result: PASS — No net-new dependencies introduced.**

---

## Known Gap — Cross-Actor Telemetry Forwarding

### Gap Description

`DiffusionPipeline.setTelemetry(_:)` does **not** forward the reporter to
`MemoryManager.shared.setTelemetry(_:)`. A consumer who follows the old single-call
README snippet will NOT observe the terminal `unloadSegment(role: nil, …)` event
emitted from `MemoryManager.clearGPUCache`. That event carries the Metal
`currentAllocatedSize` before/after delta — the canonical "did unload free GPU
pages?" probe.

### Decision: Option A — Documentation Fix

**Chosen option: A (documentation update)**

The README and AGENTS.md consumer snippets have been updated to show **two** `setTelemetry`
calls: one on the pipeline, one on `MemoryManager.shared`.

**Rationale:**

1. `MemoryManager.shared` is a **global actor singleton**. If `DiffusionPipeline.setTelemetry`
   auto-forwarded to the singleton, any pipeline (including multiple sequential test-pipeline
   instances) would silently overwrite the singleton's telemetry state in a last-writer-wins
   fashion. The existing Sortie 10 tests (`MemoryManagerTelemetryTests`) explicitly manage
   `MemoryManager.shared.setTelemetry` independently — auto-forwarding would make those tests
   non-deterministic.

2. The existing seam setters on `DiffusionPipeline` (`setComponentReadinessService`,
   `setMemoryGate`) do not forward to `MemoryManager.shared`. Auto-forwarding `setTelemetry`
   would be the only cross-actor forwarding setter and would break the established convention.

3. Option A's cost is low: four lines in README, five lines in AGENTS.md. The explicit
   two-call pattern is honest about what the library actually does and gives consumers the
   flexibility to attach different reporters to the pipeline and to the memory manager
   (e.g., verbose memory reporter on `MemoryManager.shared`, filtered reporter on the pipeline).

**Files modified (Option A):**
- `README.md` — expanded consumer snippet to show both `pipeline.setTelemetry(reporter)` and
  `MemoryManager.shared.setTelemetry(reporter)` with inline comments explaining each.
- `AGENTS.md` — same update in the "Opt-in contract" section under `## Telemetry`.

No Swift source files were modified for this gap resolution.

---

## Ship-Readiness Verdict

**READY TO SHIP as 0.7.0**

All §12 acceptance criteria pass. 81/81 tests pass. No new dependencies. Public API is
purely additive (3 new types, 2 new setters, 2 new defaulted parameters). Existing callers
(`pixart-swift-mlx`, `flux-2-swift-mlx`) recompile without source changes.

One known deviation from the original 12-sortie plan (accepted during Sortie 6):
four `throw PipelineError` sites inside `private static func validateAssembly` are
un-instrumentable — that function is static-sync and runs at init time, before any consumer
can call `setTelemetry`. The 14 `errorThrown` emits that were wired cover all public-API
reachable throw sites. This deviation is documented in SUPERVISOR_STATE.md and does not
affect observable correctness.

One architectural note for future contributors: `MemoryManager.shared.setTelemetry` must
be called separately from `DiffusionPipeline.setTelemetry` to observe the Metal-bytes
probe. The consumer API surface now documents this explicitly (see README.md § Telemetry
and AGENTS.md § Telemetry / Opt-in contract).
