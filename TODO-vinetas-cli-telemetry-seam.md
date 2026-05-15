# RESOLVED — Process-wide Telemetry Seam for CLI Hosts
<!-- RESOLVED in commit: see development branch — telemetry: add process-wide TuberiaTelemetry seam for CLI hosts -->

**Filed by**: SwiftVinetas — OPERATION WIRETAP DARKROOM (2026-05-15)
**Issue surfaced in**: `SwiftVinetas/Sources/VinetasCLICore/Telemetry/CLITelemetryBootstrap.swift`

## Background

The SwiftVinetas CLI host (`vinetas generate --telemetry …`) wants to install a `TuberiaTelemetryReporter` adapter so that Tuberia events (the diffusion pipeline plumbing used by both Flux2 and PixArt under the hood) are interleaved into the unified JSONL trace.

`DiffusionPipeline.setTelemetry(_:)` at `Sources/Tuberia/Pipeline/DiffusionPipeline+Telemetry.swift:31` is **instance-bound** — each `DiffusionPipeline` is owned privately by a SwiftVinetas engine actor (`Flux2Engine`, `PixArtEngine`). The CLI bootstrap has no reference to those instances. **Today, no Tuberia events reach the CLI.**

This is the highest-volume telemetry source by case count — Tuberia ships **27** event cases covering text-encoder forward passes, scheduler config, denoise loop, backbone forward, decoder decode, renderer render, LoRA apply/unapply, weight load, and more. Losing it gives up the bulk of the per-generation observability.

## What would unblock the CLI

1. **Process-wide reporter shared by all `DiffusionPipeline` instances.** Add `public static var TuberiaTelemetry.shared: (any TuberiaTelemetryReporter)?`, have every emission site read it lazily on each call. The CLI bootstrap assigns once at startup; both Flux2 and PixArt paths get covered automatically because both ride Tuberia. Matches `SwiftAcervo.AcervoManager.shared`.
2. **Per-instance install API exposed through SwiftVinetas** — would require `Flux2Engine` and `PixArtEngine` to grow `public func setTuberiaDepReporter(_:)` and forward to their pipeline. Workable but means every host has to do per-instance wiring.

Recommendation: option (1) — Tuberia is the natural place for a process-wide seam because it's the single library all dep diffusion pipelines flow through.

## Out of scope for this TODO

- Keep the existing instance-bound `setTelemetry(_:)` — it's the right primitive. The ask is an additive process-wide layer.
- Don't change the event enum or the reporter protocol — SwiftVinetas already ships an adapter (`TuberiaTelemetryCLIAdapter`) conforming to the existing protocol.

## What's already shipped on the SwiftVinetas side

- `Sources/VinetasCLICore/Telemetry/TuberiaEventEncoding.swift` — Encodable shim for all 27 cases of `TuberiaTelemetryEvent`.
- `Sources/VinetasCLICore/Telemetry/TuberiaTelemetryCLIAdapter.swift` — conforms to `TuberiaTelemetryReporter`, writes with `kind: "tuberia"`.

When the process-wide seam lands, the CLI just calls `<NewSeam>.setReporter(adapter)` in one line and the integration test's `kinds ⊇ {tuberia}` assertion will start passing.
