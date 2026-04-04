# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For detailed project documentation, see **[AGENTS.md](AGENTS.md)**.

## Quick Reference

**Project**: SwiftTuberia - Componentized generation pipeline for MLX inference

**Platforms**: iOS 26.0+, macOS 26.0+

**Key Components**:
- Pipe segment protocols: TextEncoder, Scheduler, Backbone, Decoder, Renderer
- DiffusionPipeline compositor with PipelineRecipe protocol
- Shared component catalog (T5-XXL, SDXL VAE, DPM-Solver++, FlowMatch Euler)
- Infrastructure: WeightLoader, MemoryManager, LoRA, Progress

**Important Notes**:
- ONLY supports iOS 26.0+ and macOS 26.0+ (NEVER add code for older platforms)
- Two products: `Tuberia` (protocols + infra) and `TuberiaCatalog` (concrete components)
- See [AGENTS.md](AGENTS.md) for complete documentation
- See [REQUIREMENTS.md](REQUIREMENTS.md) for full specification

## Claude-Specific Build Preferences

- NEVER use `swift build` or `swift test` — always use `xcodebuild`
- Use XcodeBuildMCP tools (`swift_package_build`, `swift_package_test`) when available
- See global `~/.claude/CLAUDE.md` for MCP server configuration and communication preferences
