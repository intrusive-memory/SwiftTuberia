import Foundation
import SwiftAcervo
import Tuberia

// MARK: - Acervo Component Registration

/// T5-XXL Encoder component descriptor.
///
/// Registered at module initialization so the Acervo Component Registry
/// is populated before any model loading or download is attempted.
private let t5XXLEncoderComponentDescriptor = SwiftAcervo.ComponentDescriptor(
  id: "t5-xxl-encoder-int4",
  type: .encoder,
  displayName: "T5-XXL Text Encoder (int4)",
  repoId: "intrusive-memory/t5-xxl-int4-mlx",
  minimumMemoryBytes: 2_000_000_000,
  metadata: [
    "component_role": "text_encoder",
    "quantization": "int4",
    "output_dim": "4096",
    "max_sequence_length": "512",
  ]
)

/// SDXL VAE Decoder component descriptor.
///
/// Registered at module initialization so the Acervo Component Registry
/// is populated before any model loading or download is attempted.
private let sdxlVAEDecoderComponentDescriptor = SwiftAcervo.ComponentDescriptor(
  id: "sdxl-vae-decoder-fp16",
  type: .decoder,
  displayName: "SDXL VAE Decoder (fp16)",
  repoId: "intrusive-memory/sdxl-vae-fp16-mlx",
  minimumMemoryBytes: 500_000_000,
  metadata: [
    "component_role": "decoder",
    "quantization": "fp16",
    "latent_channels": "4",
  ]
)

/// Module-level registration trigger.
///
/// This `let` is evaluated once (lazily) on first access, registering all
/// TuberiaCatalog component descriptors with the SwiftAcervo Component Registry.
private let _registerTuberiaCatalogComponents: Void = {
  Acervo.register([
    t5XXLEncoderComponentDescriptor,
    sdxlVAEDecoderComponentDescriptor,
  ])
}()

// MARK: - CatalogRegistration

/// Public interface to TuberiaCatalog's component registration and discovery.
///
/// This class ensures Acervo components are registered and provides utility
/// methods for component lookup. The actual registration is handled by the
/// private `_registerTuberiaCatalogComponents` trigger.
public final class CatalogRegistration: @unchecked Sendable {

  /// Singleton instance.
  public static let shared = CatalogRegistration()

  private init() {}

  // MARK: - Component Descriptors (Public for convenience)

  /// T5-XXL Encoder (int4 quantized) component ID.
  public static let t5XXLEncoderComponentId = "t5-xxl-encoder-int4"

  /// SDXL VAE Decoder (fp16) component ID.
  public static let sdxlVAEDecoderComponentId = "sdxl-vae-decoder-fp16"

  // MARK: - Ensure Registration

  /// Ensure all catalog components are registered.
  /// Safe to call multiple times -- only performs registration on first call.
  public func ensureRegistered() {
    _ = _registerTuberiaCatalogComponents
  }

  // MARK: - Queries (delegated to Acervo)

  /// Look up a component descriptor by its ID.
  public func descriptor(for componentId: String) -> SwiftAcervo.ComponentDescriptor? {
    Acervo.component(componentId)
  }

  /// Check if a component is registered.
  public func isComponentRegistered(_ componentId: String) -> Bool {
    Acervo.component(componentId) != nil
  }

  /// Ensure a component is downloaded and ready.
  ///
  /// - Parameter componentId: The component to prepare.
  /// - Throws: AcervoError if download or validation fails.
  @available(*, deprecated, message: "Use ComponentReadinessService (with progress) or Acervo.ensureComponentReady directly.")
  public func ensureComponentReady(_ componentId: String) async throws {
    try await Acervo.ensureComponentReady(componentId)
  }
}
