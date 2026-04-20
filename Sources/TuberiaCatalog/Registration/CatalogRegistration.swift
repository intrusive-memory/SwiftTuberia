import Foundation
import SwiftAcervo
import Tuberia

// MARK: - Required Component Files

/// T5-XXL Encoder (int4 quantized) required files.
///
/// The model is hosted on HuggingFace at `intrusive-memory/t5-xxl-int4-mlx`.
/// All safetensors shards, configuration, and tokenizer files are required for
/// both weight loading and tokenization.
///
/// SHA-256 digests and sizes verified from local SharedModels group container
/// on 2026-04-20 (S2 REQ-T4). The model uses 5 sharded safetensors shards.
private let t5XXLEncoderRequiredFiles: [ComponentFile] = [
  ComponentFile(
    relativePath: "config.json",
    expectedSizeBytes: 2150,
    sha256: "dbb4d38c53a37b5cc2bb67262eb56b776f2d828f88189788ea120b39b189a826"
  ),
  ComponentFile(
    relativePath: "tokenizer.json",
    expectedSizeBytes: 2_424_235,
    sha256: "f5dfec163765e18e270537fe896c49f5fad74db1525641d9b255a3008b999596"
  ),
  ComponentFile(
    relativePath: "tokenizer_config.json",
    expectedSizeBytes: 20_848,
    sha256: "d8e1edceb843032e85dcf4f7736fbb224b4ab0ef3e8c2259e858d07f67df99af"
  ),
  ComponentFile(
    relativePath: "special_tokens_map.json",
    expectedSizeBytes: 2543,
    sha256: "7a1985a994c41886db38c719d2a3d2f40606663cc19d7c5d6a85d349320e06d2"
  ),
  ComponentFile(
    relativePath: "model-00000-of-00005.safetensors",
    expectedSizeBytes: 715_258_584,
    sha256: "6657ed6942a268a7d954029b3cabac5677547a847eead1d86ea9eda9b4d17b68"
  ),
  ComponentFile(
    relativePath: "model-00001-of-00005.safetensors",
    expectedSizeBytes: 733_612_624,
    sha256: "628389d52839ddc967558f68999d37a42c8b9b2fc6c6e6858535bc30da837c3e"
  ),
  ComponentFile(
    relativePath: "model-00002-of-00005.safetensors",
    expectedSizeBytes: 732_555_480,
    sha256: "8c3e0e32220643ac8a7b1cd4c170455f876055df9c0acd48dd8fb9fc7a9525d7"
  ),
  ComponentFile(
    relativePath: "model-00003-of-00005.safetensors",
    expectedSizeBytes: 497_732_304,
    sha256: "1e78f2aebf12dac654f754276f18c3c91fd0d17a416ca6a60c0ff137fb68cfef"
  ),
  ComponentFile(
    relativePath: "model-00004-of-00005.safetensors",
    expectedSizeBytes: 263_192_672,
    sha256: "55b3ab4c70040390f2a8eef938e0bd93a6543e2e33e687f5d077f051723963f7"
  ),
]

/// SDXL VAE Decoder (fp16) required files.
///
/// The model is hosted on HuggingFace at `intrusive-memory/sdxl-vae-fp16-mlx`.
/// Includes the model weights and configuration for decoding.
///
/// SHA-256 digests and sizes verified from local SharedModels group container
/// on 2026-04-20 (S2 REQ-T4).
private let sdxlVAEDecoderRequiredFiles: [ComponentFile] = [
  ComponentFile(
    relativePath: "config.json",
    expectedSizeBytes: 1054,
    sha256: "fa21368160b774bfaf32f7b0999912fcd79269a3fdb14b0a79698d45bd42dcfd"
  ),
  ComponentFile(
    relativePath: "model.safetensors",
    expectedSizeBytes: 167_335_310,
    sha256: "0e636ee29d502d344094f3c03624b462b7e37efb68c61f1f1d47495ef6a0d2db"
  ),
]

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
  files: t5XXLEncoderRequiredFiles,
  estimatedSizeBytes: 1_288_490_188,  // ~1.2 GB
  minimumMemoryBytes: 2_000_000_000,
  metadata: [
    "component_role": "text_encoder",
    "quantization": "int4",
    "output_dim": "4096",
    "max_sequence_length": "512"
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
  files: sdxlVAEDecoderRequiredFiles,
  estimatedSizeBytes: 167_772_160,  // ~160 MB
  minimumMemoryBytes: 500_000_000,
  metadata: [
    "component_role": "decoder",
    "quantization": "fp16",
    "latent_channels": "4"
  ]
)

/// Module-level registration trigger.
///
/// This `let` is evaluated once (lazily) on first access, registering all
/// TuberiaCatalog component descriptors with the SwiftAcervo Component Registry.
private let _registerTuberiaCatalogComponents: Void = {
  Acervo.register([
    t5XXLEncoderComponentDescriptor,
    sdxlVAEDecoderComponentDescriptor
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
  public func ensureComponentReady(_ componentId: String) async throws {
    try await Acervo.ensureComponentReady(componentId)
  }
}
