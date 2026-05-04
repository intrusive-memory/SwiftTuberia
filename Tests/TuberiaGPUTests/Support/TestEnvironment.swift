import Foundation

/// Bootstraps the SwiftAcervo App Group identifier for GPU-test runs. See
/// the matching helper in `TuberiaTests/Support/TestEnvironment.swift` for
/// the full rationale; duplicated here because Swift test targets cannot
/// share helper sources directly.
enum TestEnvironment {

  static let acervoAppGroupEnvVar = "ACERVO_APP_GROUP_ID"
  static let defaultTestAppGroupID = "group.intrusive-memory.tuberia.tests"

  static func ensureAcervoAppGroup() {
    _ = bootstrap
  }

  private static let bootstrap: Void = {
    if let existing = ProcessInfo.processInfo.environment[acervoAppGroupEnvVar],
      !existing.isEmpty
    {
      return
    }
    setenv(acervoAppGroupEnvVar, defaultTestAppGroupID, 1)
  }()
}
