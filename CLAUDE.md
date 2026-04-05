# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read [AGENTS.md](AGENTS.md) first** for universal project documentation (architecture, components, build commands, critical rules).

## Claude-Specific Build Preferences

- NEVER use `swift build` or `swift test` — always use `xcodebuild`
- Use XcodeBuildMCP tools (`swift_package_build`, `swift_package_test`) when available
- Always include `arch=arm64` in every xcodebuild `-destination` (Apple Silicon only)
- See global `~/.claude/CLAUDE.md` for MCP server configuration and communication preferences
