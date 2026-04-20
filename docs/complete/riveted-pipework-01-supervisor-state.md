# SUPERVISOR_STATE.md

> **Terminology**: *Mission* = definable scope of work. *Sortie* = atomic agent task.

## Mission Metadata

- **Operation**: OPERATION RIVETED PIPEWORK
- **Iteration**: 1
- **Mission branch**: `mission/riveted-pipework/01`
- **Starting-point commit**: `dcc1eec00e96125c18596b110cd73974e338ff4b`
- **Plan**: `EXECUTION_PLAN.md`
- **Initialized**: 2026-04-20
- **Max retries per sortie**: 3

## Plan Summary

- Work units: 1 (SwiftTuberia-v2 at project root)
- Total sorties: 8
- Dependency structure: layered DAG (4 layers; S1/S2/S3 parallel-eligible in Layer 1)
- Dispatch mode: dynamic

## Work Units

| Name | Directory | Sorties | Dependencies |
|------|-----------|---------|--------------|
| SwiftTuberia-v2 | . | 8 | none |

## Sortie Dependency Graph

| Sortie | ID | Layer | Depends On | State | Attempt |
|--------|----|-------|------------|-------|---------|
| 1 | REQ-T5 | 1 | — | COMPLETED | 1/3 |
| 2 | REQ-T4 | 1 | — | COMPLETED¹ | 1/3 |
| 3 | REQ-PIPE-01 | 1 | — | COMPLETED | 1/3 |
| 4 | REQ-PIPE-02 | 2 | S3 ✅, S5 ✅ | COMPLETED | 1/3 |
| 5 | REQ-PIPE-03 | 2 | S3 ✅ | COMPLETED | 1/3 |
| 6 | REQ-INT-01 | 2 | S2 ✅ | COMPLETED | 1/3 |
| 7 | REQ-CDN-01 | 2 | S2 ✅ | COMPLETED | 1/3 |
| 8 | REQ-DOC-01 | 4 | S1–S7 ✅ | COMPLETED | 1/3 |

## Work Unit State

### SwiftTuberia-v2
- Work unit state: COMPLETED
- All 8 sorties committed on `mission/riveted-pipework/01`
- Last verified: S8 commit de072ec; all exit criteria met with noted caveats (S2 count deviation from plan, S8 grep-pattern bug in plan exit criterion)
- Final commit graph: 0aa8fcf (S1) → de8212c (S3) → dc88d6d (S2) → 405168e (S5) → 0c58bf5 (S4) → bf761d0 (S6) → f4e6939 (S7) → de072ec (S8)

## Active Agents

| Work Unit | Sortie | Sortie State | Attempt | Model | Complexity Score | Task ID | Output File | Dispatched At |
|-----------|--------|--------------|---------|-------|------------------|---------|-------------|---------------|
| SwiftTuberia-v2 | S1 (REQ-T5) | COMPLETED | 1/3 | haiku | 4 | a8ff160792020114e | — | 2026-04-20 → commit 0aa8fcf |
| SwiftTuberia-v2 | S2 (REQ-T4) | COMPLETED¹ | 1/3 | sonnet | 9 | a89e61bfd2ea0475a | — | 2026-04-20 → commit dc88d6d |
| SwiftTuberia-v2 | S5 (REQ-PIPE-03) | COMPLETED | 1/3 | sonnet | 10 | aab362dcb320e2183 | — | 2026-04-20 → commit 405168e |
| SwiftTuberia-v2 | S4 (REQ-PIPE-02) | COMPLETED | 1/3 | sonnet | 8 | aa80bafd64016fdd7 | — | 2026-04-20 → commit 0c58bf5 |
| SwiftTuberia-v2 | S6 (REQ-INT-01) | COMPLETED | 1/3 | sonnet | 11 | aff3a1957f6b0347e | — | 2026-04-20 → commit bf761d0 |
| SwiftTuberia-v2 | S7 (REQ-CDN-01) | COMPLETED | 1/3 | sonnet | 9 | a65f565a1e163b961 | — | 2026-04-20 → commit f4e6939 |
| SwiftTuberia-v2 | S8 (REQ-DOC-01) | COMPLETED | 1/3 | sonnet | 6 | aabcd0ad49f4cb0d7 | — | 2026-04-20 → commit de072ec |
| SwiftTuberia-v2 | S3 (REQ-PIPE-01) | COMPLETED | 1/3 | sonnet | 12 | a7065f8e410adb4d4 | — | 2026-04-20 → commit de8212c |

## Decisions Log

| Timestamp | Work Unit | Sortie | Decision | Rationale |
|-----------|-----------|--------|----------|-----------|
| 2026-04-20T00:00Z | — | — | Operation named OPERATION RIVETED PIPEWORK | THE RITUAL (haiku-class pattern generation, executed inline) |
| 2026-04-20T00:00Z | — | — | Iteration set to 1 | No `*_BRIEF.md` files in project root |
| 2026-04-20T00:00Z | SwiftTuberia-v2 | S1 | Model: haiku | Trivial single-version bump + resolve + AGENTS.md entry; complexity score ~4 |
| 2026-04-20T00:00Z | SwiftTuberia-v2 | S2 | Model: sonnet | CDN download + shasum + 6 descriptor rewrites + new test suite; complexity ~10 |
| 2026-04-20T00:00Z | SwiftTuberia-v2 | S3 | Model: sonnet | Foundation sortie for S4/S5; protocol-backed seam + test double + progress threading; complexity ~11 |
| 2026-04-20T00:00Z | SwiftTuberia-v2 | Layer 1 | Dispatch S1, S2, S3 in parallel | DAG shows no inter-dependency; each touches distinct files |

## Notes

- Uncommitted changes carried onto this branch from `development`: `M EXECUTION_PLAN.md`, `M requirements/INFRASTRUCTURE.md`, `?? REQUIREMENTS.md`. These are intentional scope artifacts (the plan + requirements that describe this mission). Sorties may commit them as they touch those files.
- Build constraint: all sorties with `xcodebuild` in exit criteria (S1–S6) must run the builds themselves. Agents dispatched with full tool access.

## Pre-existing Issues Surfaced (Not In Scope)

- **Metal GPU test crash**: S6 ran the full `xcodebuild test` suite and reported `T5EncodeWithSyntheticWeightsTests.encodeOutputIsNonZero` hits a Metal GPU crash. S6 verified it predates this mission by stashing changes and reproducing on HEAD. Impact on this mission: S8's Master Acceptance check #2 (`xcodebuild test ... passes on a clean clone`) will fail. Out of scope to fix here; flag in mission brief.

## Deviations & Supervisor Errors

**¹ S2 exit-criteria deviation (non-blocking, plan defect)**: Plan asserted `grep -cE 'sha256: ...'` == 6 based on an assumption that `model.safetensors` is a single file for T5. Reality: `intrusive-memory_t5-xxl-int4-mlx` is sharded across 5 safetensors files + 4 metadata files = 9 T5 files; combined with 2 VAE files = 11 total. Actual count is 11 ≠ 6, but the INTENT of the exit criterion (every ComponentFile has sha256+expectedSizeBytes) is satisfied:
  - `grep -nE 'ComponentFile\([^)]*\)' ... | grep -v 'sha256:' | wc -l` = 0 (every entry has sha256). PASS.
  - `xcodebuild test` passed including the new ComponentFileIntegrityTests suite.
  - Treating as COMPLETED; planned-count exit criterion marked as a plan defect not an implementation gap. REQUIREMENTS.md / EXECUTION_PLAN.md should be updated at brief time to reflect actual T5 sharding.

**¹ S2 data-source deviation (warrants user attention)**: CDN returned 404 for all canonical artifacts; the SHA-256 digests were sourced from the local group container at `group.intrusive-memory.models/SharedModels`, not the CDN. Implication: when CI first runs `ensure-model-cdn.yml` and uploads, the resulting manifest must match these local checksums byte-for-byte. S7's manifest verifier will surface any divergence — that's its purpose. But a future CDN push with a different T5 quantization would silently break S7's gate.

**Supervisor error (dispatch ordering)**: I dispatched S5 after S3 completed but while S2 was still running. The plan explicitly sequences Layer 2 (which includes S5) AFTER all of Layer 1 (which includes S2). S2's recovery revert of `DiffusionPipeline.swift` to HEAD destroyed S5's uncommitted working-tree edits. S5's agent appears to have re-applied them (working tree currently shows S5 edits on all three target files), but this was luck, not design. Going forward: single active agent for the remaining sorties to eliminate working-tree contention. No worktree isolation was configured.
