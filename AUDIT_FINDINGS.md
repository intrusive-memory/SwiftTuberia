---
title: "SwiftTuberia — Audit-Aligned Sorties (Acervo v2 Integration)"
date: 2026-04-18
source: "ACERVO_CONSUMER_AUDIT.md (lines 139–157)"
version: "1.0"
status: "AUDIT FINDINGS — SORTIES READY"
---

# SwiftTuberia — Audit-Aligned Sorties

**Priority**: 🔴 **HIGHEST**  
**Mission Status**: Duplicate registry, mixed APIs, missing integrity verification  
**Scope**: Fix TuberiaCatalog integration with SwiftAcervo v2 API

---

## Audit Summary

SwiftTuberia currently uses a **duplicate `ComponentDescriptor` type** and a **custom `CatalogRegistration` pattern** instead of SwiftAcervo's standard `Acervo.register()` API. The WeightLoader uses v1 API (`withModelAccess()`) and TuberiaCatalog descriptors lack SHA-256 checksums.

| Aspect | Status | Details |
|--------|--------|---------|
| Component Registration | ❌ DUPLICATE | Defines own `ComponentDescriptor` in CatalogRegistration (should use SwiftAcervo's) |
| Registry Integration | ❌ WRONG | Uses custom `CatalogRegistration` instead of `Acervo.register()` |
| v1 API Usage | ✅ ACTIVE | `WeightLoader` uses `AcervoManager.shared.withModelAccess()` |
| v2 API Adoption | ❌ MISSING | Should migrate to `withComponentAccess()` |
| Integrity Verification | ❌ MISSING | TuberiaCatalog ComponentDescriptor has no checksums |

**Master Index**: See [`/Users/stovak/Projects/REQUIREMENTS.md`](/Users/stovak/Projects/REQUIREMENTS.md) for mission context, terminology, and cross-project tracking.

**Architectural Context**: See [`REQUIREMENTS.md`](REQUIREMENTS.md) for SwiftTuberia's design philosophy and the plumbing system metaphor.

---

## Sorties

### Sortie T1: Remove Duplicate ComponentDescriptor Type

**Priority**: 🔴 CRITICAL  
**Owner**: (unassigned)  
**Status**: PENDING

**Description**:
Remove the custom `ComponentDescriptor` type defined in `CatalogRegistration.swift`. This duplicates SwiftAcervo's canonical type and prevents using `Acervo.register()`.

**Entry Criteria**:
- [ ] File location identified: `CatalogRegistration.swift` in Tuberia target
- [ ] Type definition located and reviewed
- [ ] All usages of custom type inventoried

**Work**:
1. Delete custom `ComponentDescriptor` struct from `CatalogRegistration.swift`
2. Update all internal references to use `import SwiftAcervo` + `SwiftAcervo.ComponentDescriptor`
3. Verify no other files define their own ComponentDescriptor

**Exit Criteria**:
- [ ] Custom type removed
- [ ] Code compiles with `swift_package_build`
- [ ] All catalog component definitions updated
- [ ] Tests pass: `swift_package_test`

---

### Sortie T2: Migrate to Acervo.register() Pattern

**Priority**: 🔴 CRITICAL  
**Owner**: (unassigned)  
**Status**: PENDING

**Description**:
Replace custom `CatalogRegistration` pattern with SwiftAcervo's standard `Acervo.register()` API. This is the canonical way to register components at module init.

**Entry Criteria**:
- [ ] Sortie T1 (remove duplicate type) complete
- [ ] Current `CatalogRegistration` calls inventoried across codebase
- [ ] SwiftAcervo 0.7.1+ dependency confirmed in Package.swift

**Work**:
1. Locate all `CatalogRegistration` call sites (likely in TuberiaCatalog's module init or component factory)
2. Replace each with `Acervo.register(descriptor: ...)` using SwiftAcervo types
3. Ensure registration happens at module load time (before first component access)
4. Remove or refactor `CatalogRegistration.swift` to simple factory if needed

**Exit Criteria**:
- [ ] All registrations use `Acervo.register()`
- [ ] Code compiles and tests pass
- [ ] No lingering `CatalogRegistration` references
- [ ] Descriptor fields match SwiftAcervo's ComponentDescriptor schema

---

### Sortie T3: Update WeightLoader to v2 API

**Priority**: 🔴 CRITICAL  
**Owner**: (unassigned)  
**Status**: PENDING

**Description**:
Update `WeightLoader` to use `AcervoManager.shared.withComponentAccess()` instead of the deprecated `withModelAccess()`. This is the v2 API required for integrity-verified model access.

**Entry Criteria**:
- [ ] Sortie T2 (Acervo.register() migration) complete
- [ ] WeightLoader.swift located and reviewed
- [ ] All `withModelAccess()` call sites identified
- [ ] SwiftAcervo 0.7.1+ documentation available for `withComponentAccess()` signature

**Work**:
1. Find all `withModelAccess()` calls in WeightLoader
2. Replace each with `withComponentAccess()` closure
3. Update closure signature: `(Component) -> T` instead of `(ModelPath) -> T`
4. Verify checksums are validated automatically by Acervo framework (no manual checksum code needed)

**Exit Criteria**:
- [ ] All `withModelAccess()` calls replaced
- [ ] Closure signatures match v2 API
- [ ] Code compiles and tests pass
- [ ] Weight loading integration tests validate component checksums

---

### Sortie T4: Add SHA-256 Checksums to TuberiaCatalog Descriptors

**Priority**: 🔴 CRITICAL  
**Owner**: (unassigned)  
**Status**: PENDING

**Description**:
Add SHA-256 checksums to all component descriptors in TuberiaCatalog (T5XXLEncoder, SDXLVAEDecoder, etc.). Checksums enable integrity verification and are required by SwiftAcervo v2.

**Entry Criteria**:
- [ ] Sortie T3 (WeightLoader v2 API) complete
- [ ] TuberiaCatalog.swift (or equivalent catalog definitions) located
- [ ] List of all catalog components: T5XXLEncoder, SDXLVAEDecoder, DPMSolverScheduler, FlowMatchEulerScheduler, ImageRenderer, AudioRenderer
- [ ] Upstream model sources identified (HuggingFace, MLX artifacts)

**Work**:
1. For each catalog component, locate the HuggingFace repo or upstream source
2. Download the canonical model artifact (safetensors, config.json, etc.)
3. Generate SHA-256 checksums: `shasum -a 256 <file> | awk '{print $1}'`
4. Update each ComponentDescriptor with `checksums: [.sha256("...")]`
5. Document upstream source and version in descriptor comments

**Exit Criteria**:
- [ ] All catalog descriptors have checksums
- [ ] Checksums verified against actual artifacts
- [ ] Code compiles and tests pass
- [ ] Integration test validates that AcervoManager rejects mismatched checksums

---

### Sortie T5: Update SwiftAcervo Dependency Version

**Priority**: 🔵 HIGH  
**Owner**: (unassigned)  
**Status**: PENDING

**Description**:
Update `Package.swift` to require SwiftAcervo 0.7.1 or later (supports v2 API: `withComponentAccess()`, checksums, component descriptors).

**Entry Criteria**:
- [ ] Sorties T1–T4 complete
- [ ] SwiftAcervo 0.7.1+ release published to GitHub
- [ ] Release notes confirm v2 API stability

**Work**:
1. Edit `Package.swift` dependency section: `.package(url: "...", from: "0.7.1")`
2. Run `swift package resolve` to fetch new version
3. Confirm no breaking changes in our integration

**Exit Criteria**:
- [ ] Package.swift updated
- [ ] Swift package resolves cleanly
- [ ] Tests pass: `swift_package_test`

---

## Non-Blocking Items (Deferred)

These items are related but do not block the main audit sorties:

- **LoRA weight loading**: WeightLoader may need LoRA-specific checksum validation; defer until LoRA support is implemented in DiffusionPipeline
- **Custom component registration**: Plugins (pixart-swift-mlx, etc.) will register their own backbones; ensure they follow same pattern once T1–T2 complete
- **Renderer integrity**: AudioRenderer and ImageRenderer are stateless; checksums only needed if they access external assets

---

## Integration Checkpoints

### When T1–T2 Are Complete:
- All custom types removed
- Catalog components register via `Acervo.register()`
- Code compiles and links to SwiftAcervo types cleanly

### When T1–T4 Are Complete:
- WeightLoader uses v2 API with integrity checking
- All component descriptors have checksums
- Integration test confirms: AcervoManager enforces checksums on access
- Minimum SwiftAcervo version bumped to 0.7.1+

### Success Criteria (All Sorties Complete):
- [ ] Code compiles: `swift_package_build`
- [ ] All tests pass: `swift_package_test`
- [ ] Model plugins (pixart-swift-mlx, etc.) still link and load weights successfully
- [ ] Audit checklist updated in master REQUIREMENTS.md

---

## References

- **Audit Source**: `/Users/stovak/Projects/ACERVO_CONSUMER_AUDIT.md` (lines 139–157)
- **Master Requirements**: `/Users/stovak/Projects/REQUIREMENTS.md`
- **Architectural Spec**: `/Users/stovak/Projects/SwiftTuberia/REQUIREMENTS.md` (plumbing system, protocols, pipeline design)
- **SwiftAcervo Documentation**: SwiftAcervo repo (PROTOCOLS.md, INFRASTRUCTURE.md)

---

## History

| Date | Event |
|------|-------|
| 2026-04-18 | Audit findings documented; audit-aligned sorties created in AUDIT_FINDINGS.md |
