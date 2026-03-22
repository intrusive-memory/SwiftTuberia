# Contributing to SwiftTuberia

Thank you for your interest in contributing to SwiftTuberia. This document covers the development setup, testing guidelines, commit conventions, and pull request process.

## Development Setup

### Prerequisites

| Requirement | Minimum Version |
|------------|----------------|
| macOS      | 26.0+          |
| Xcode      | 26+            |
| Swift      | 6.2+           |

### Clone and Build

```bash
git clone https://github.com/intrusive-memory/SwiftTuberia.git
cd SwiftTuberia
xcodebuild build -scheme SwiftTuberia -destination 'platform=macOS'
```

## Testing

### Running Unit Tests

```bash
xcodebuild test -scheme SwiftTuberia -destination 'platform=macOS'
```

### Test Guidelines

- All new public API methods must have corresponding unit tests.
- Unit tests must not require network access or model downloads.
- Tests must pass on both macOS and iOS Simulator targets.

## Commit Conventions

Use clear, imperative-mood commit messages:

### Prefixes

- **Add** -- A wholly new feature or file.
- **Update** -- An enhancement to existing functionality.
- **Fix** -- A bug fix.
- **Remove** -- Removal of code, files, or features.
- **Refactor** -- Code restructuring with no behavior change.
- **Test** -- Adding or updating tests only.
- **Docs** -- Documentation-only changes.

### Rules

- Keep the first line under 72 characters.
- Use the body for additional context when the change is non-obvious.

## Pull Request Process

1. **Branch from `development`.** Create a feature branch off `development`, not `main`.
2. **Keep changes focused.** Each PR should address a single concern.
3. **Ensure tests pass.** Run tests locally before opening a PR.
4. **Open the PR against `development`.**
5. **CI must pass.** GitHub Actions runs tests on macOS and iOS Simulator.

## Code Style

- Follow Swift API Design Guidelines.
- Use `///` doc comments for all public API.
- All closures in public API must be `@Sendable` (Swift 6 strict concurrency).

## Platform Requirements

SwiftTuberia targets iOS 26.0+ and macOS 26.0+ exclusively. Do not add `@available` or `#available` checks for older platform versions.
