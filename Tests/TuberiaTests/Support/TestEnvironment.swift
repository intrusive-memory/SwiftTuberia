import Foundation

/// Bootstraps the SwiftAcervo App Group identifier for unit-test runs.
///
/// SwiftAcervo 0.10+ calls `fatalError` from `Acervo.sharedModelsDirectory`
/// when no App Group identifier is configured. CLI tools, scripts, and CI
/// runners satisfy this by exporting `ACERVO_APP_GROUP_ID` in the shell
/// (typically `~/.zprofile`). Tests must do the same before exercising any
/// code path that resolves the shared models directory.
///
/// Resolution order:
/// 1. If `ACERVO_APP_GROUP_ID` is already set in the environment (e.g. from
///    the developer's shell), leave it alone.
/// 2. Otherwise set a stable test-only group identifier so CI runners that
///    do not export the variable can still resolve a writable path.
///
/// Mirrors the per-test `setenv` pattern used in SwiftProyecto's
/// `AcervoDownloadIntegrationTests`.
enum TestEnvironment {

  static let acervoAppGroupEnvVar = "ACERVO_APP_GROUP_ID"
  static let defaultTestAppGroupID = "group.intrusive-memory.tuberia.tests"

  /// Idempotent. Safe to call from many tests; only the first call has effect.
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
