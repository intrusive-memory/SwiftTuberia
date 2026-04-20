# EXECUTION_PLAN: SwiftTuberia Pipeline Architecture (LLM-Scope: RESEARCH ONLY)

**Version**: 1.0  
**Date**: 2026-04-17  
**Status**: RESEARCH PHASE (OUT OF SCOPE — LLM FOCUSED EFFORT)  
**Requirements Source**: `REQUIREMENTS.md`

---

## Terminology

**Mission** — Understand SwiftTuberia architecture for future LLM inference integration.

**Status**: RESEARCH ONLY — SwiftTuberia adoption is deferred. This plan documents baseline understanding.

---

## Mission Overview

SwiftTuberia provides a unified, componentized pipeline for MLX inference (diffusion, auto-regressive, etc.). This is infrastructure for future work; LLM-focused adoption is deferred.

**Current Understanding**:
- ✅ SwiftTuberia provides shared pipeline components (encoders, schedulers, decoders)
- ✅ Model plugins (Flux2, PixArt) plug into this system
- ✅ Infrastructure: MemoryManager, WeightLoader, Acervo integration
- ⏳ Deferred: LLM adaptation and adoption

---

## Research Sorties (No Implementation)

| Sortie | Objective | Status |
|--------|-----------|--------|
| 1.1 | Read `requirements/PROTOCOLS.md` — pipe segment contracts | RESEARCH |
| 1.2 | Read `requirements/PIPELINE.md` — DiffusionPipeline orchestration | RESEARCH |
| 1.3 | Read `requirements/CATALOG.md` — shared component catalog | RESEARCH |
| 1.4 | Read `requirements/INFRASTRUCTURE.md` — MemoryManager, WeightLoader | RESEARCH |
| 1.5 | Summarize for future LLM adoption | RESEARCH |

---

**Status**: DEFERRED IMPLEMENTATION

**Next Trigger**: After SwiftProyecto/SwiftBruja LLM support complete. Then evaluate SwiftTuberia for unified LLM inference pipeline.

---

## Future LLM Integration (Placeholder)

When LLM adoption begins:

1. Determine if SwiftTuberia's diffusion-focused pipeline adapts to LLM inference (likely not — LLMs are auto-regressive, not diffusion-based)
2. If yes: Create LLM recipes, register LLM model plugins
3. If no: Keep existing SwiftAcervo integration for LLM models; use SwiftTuberia for image/audio only

**Recommendation**: Research suggests SwiftTuberia provides infrastructure (MemoryManager, Acervo integration) useful to LLM inference, but the "pipe segments" (encoders, schedulers, decoders) are diffusion-specific. LLMs likely continue using direct SwiftAcervo integration (as in SwiftBruja, SwiftProyecto).
