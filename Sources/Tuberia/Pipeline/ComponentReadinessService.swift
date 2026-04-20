import SwiftAcervo

/// Seam for ensuring a registered Acervo component is downloaded and verified before loading.
///
/// The production conformer (`AcervoComponentReadinessService`) delegates to
/// `Acervo.ensureComponentReady`, which is a no-op when the component is already on disk.
/// In tests, a spy conformer can be injected to verify that the pipeline calls this method
/// once per weighted segment that has a non-nil `componentId`, without touching the network.
///
/// NOTE: This protocol is intentionally minimal — it mirrors only the portion of the
/// `Acervo` static API that `DiffusionPipeline.loadModels` needs. Broader Acervo
/// orchestration stays with the callers and tooling.
public protocol ComponentReadinessService: Sendable {
  /// Ensure the named component is downloaded and integrity-verified.
  ///
  /// - Parameters:
  ///   - componentId: The Acervo component identifier.
  ///   - progress: Optional progress callback receiving `AcervoDownloadProgress`.
  func ensureComponentReady(
    _ componentId: String,
    progress: (@Sendable (AcervoDownloadProgress) -> Void)?
  ) async throws
}

/// Production implementation: delegates to `Acervo.ensureComponentReady`.
///
/// This is a no-op when the component files already exist on disk (cached path).
/// When files are missing it triggers a download through the standard Acervo CDN pipeline.
public struct AcervoComponentReadinessService: ComponentReadinessService {
  public init() {}

  public func ensureComponentReady(
    _ componentId: String,
    progress: (@Sendable (AcervoDownloadProgress) -> Void)?
  ) async throws {
    try await Acervo.ensureComponentReady(componentId, progress: progress)
  }
}
