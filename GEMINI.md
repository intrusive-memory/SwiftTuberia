# Gemini-Specific Agent Instructions

**Read [AGENTS.md](AGENTS.md) first** for universal project documentation.

This file contains instructions specific to Google Gemini agents.

## Build and Test

Use standard `xcodebuild` commands (no MCP access):

```bash
# Build
xcodebuild build -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'

# Test
xcodebuild test -scheme SwiftTuberia-Package -destination 'platform=macOS,arch=arm64'

# Resolve dependencies
xcodebuild -resolvePackageDependencies
```

## Gemini-Specific Critical Rules

1. Use standard CLI tools — no MCP server access
2. NEVER use `swift build` or `swift test` — use `xcodebuild`
3. NEVER add code targeting platforms older than iOS 26.0 / macOS 26.0
4. Apple Silicon only — always include `arch=arm64` in destinations
