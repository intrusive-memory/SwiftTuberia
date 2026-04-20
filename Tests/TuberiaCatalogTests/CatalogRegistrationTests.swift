import Foundation
import Testing
import SwiftAcervo
import Tuberia

@testable import TuberiaCatalog

@Suite("CatalogRegistration Tests", .serialized)
struct CatalogRegistrationTests {

  @Test("ensureRegistered() registers components via SwiftAcervo")
  func registersExpectedComponents() {
    let registry = CatalogRegistration.shared
    registry.ensureRegistered()

    // Verify components are registered via Acervo
    #expect(registry.isComponentRegistered("t5-xxl-encoder-int4"))
    #expect(registry.isComponentRegistered("sdxl-vae-decoder-fp16"))
  }

  @Test("Registration is idempotent -- multiple calls are safe")
  func idempotentRegistration() {
    let registry = CatalogRegistration.shared

    // Register multiple times -- should not throw
    registry.ensureRegistered()
    registry.ensureRegistered()

    // Should still be registered
    #expect(registry.isComponentRegistered("t5-xxl-encoder-int4"))
    #expect(registry.isComponentRegistered("sdxl-vae-decoder-fp16"))
  }

  @Test("T5-XXL descriptor has correct metadata")
  func t5XXLDescriptor() {
    let registry = CatalogRegistration.shared
    registry.ensureRegistered()

    let descriptor = registry.descriptor(for: CatalogRegistration.t5XXLEncoderComponentId)
    #expect(descriptor != nil)
    #expect(descriptor?.id == "t5-xxl-encoder-int4")
    #expect(descriptor?.type == .encoder)
    #expect(descriptor?.repoId == "intrusive-memory/t5-xxl-int4-mlx")
    #expect(descriptor?.estimatedSizeBytes == 1_288_490_188)  // ~1.2 GB
  }

  @Test("SDXL VAE descriptor has correct metadata")
  func sdxlVAEDescriptor() {
    let registry = CatalogRegistration.shared
    registry.ensureRegistered()

    let descriptor = registry.descriptor(for: CatalogRegistration.sdxlVAEDecoderComponentId)
    #expect(descriptor != nil)
    #expect(descriptor?.id == "sdxl-vae-decoder-fp16")
    #expect(descriptor?.type == .decoder)
    #expect(descriptor?.repoId == "intrusive-memory/sdxl-vae-fp16-mlx")
    #expect(descriptor?.estimatedSizeBytes == 167_772_160)  // ~160 MB
  }

  @Test("descriptor(for:) returns correct descriptor by ID")
  func lookupById() {
    let registry = CatalogRegistration.shared
    registry.ensureRegistered()

    let t5 = registry.descriptor(for: "t5-xxl-encoder-int4")
    #expect(t5 != nil)
    #expect(t5?.repoId == "intrusive-memory/t5-xxl-int4-mlx")

    let vae = registry.descriptor(for: "sdxl-vae-decoder-fp16")
    #expect(vae != nil)
    #expect(vae?.repoId == "intrusive-memory/sdxl-vae-fp16-mlx")
  }

  @Test("descriptor(for:) returns nil for unknown ID")
  func unknownIdReturnsNil() {
    let registry = CatalogRegistration.shared
    registry.ensureRegistered()

    let unknown = registry.descriptor(for: "nonexistent-component")
    #expect(unknown == nil)
  }

  @Test("isComponentRegistered returns true for known components")
  func isRegisteredCheck() {
    let registry = CatalogRegistration.shared
    registry.ensureRegistered()

    #expect(registry.isComponentRegistered("t5-xxl-encoder-int4"))
    #expect(registry.isComponentRegistered("sdxl-vae-decoder-fp16"))
    #expect(!registry.isComponentRegistered("unknown-component"))
  }

  @Test("Component descriptors include required files")
  func requiredFiles() {
    let registry = CatalogRegistration.shared
    registry.ensureRegistered()

    let t5 = registry.descriptor(for: "t5-xxl-encoder-int4")
    // config.json, tokenizer.json, tokenizer_config.json, special_tokens_map.json,
    // model-00000-of-00005.safetensors through model-00004-of-00005.safetensors (5 shards)
    #expect(t5?.files.count == 9)

    let vae = registry.descriptor(for: "sdxl-vae-decoder-fp16")
    #expect(vae?.files.count == 2)  // config.json, model.safetensors
  }

  @Test("Component descriptors include required files with non-empty relativePaths")
  func requiredFileRelativePaths() {
    let registry = CatalogRegistration.shared
    registry.ensureRegistered()

    let t5 = registry.descriptor(for: "t5-xxl-encoder-int4")
    let vae = registry.descriptor(for: "sdxl-vae-decoder-fp16")
    for file in (t5?.files ?? []) + (vae?.files ?? []) {
      #expect(!file.relativePath.isEmpty, "ComponentFile relativePath must not be empty")
    }
  }

  @Test("Component descriptors include metadata")
  func componentMetadata() {
    let registry = CatalogRegistration.shared
    registry.ensureRegistered()

    let t5 = registry.descriptor(for: "t5-xxl-encoder-int4")
    #expect(t5?.metadata["component_role"] == "text_encoder")
    #expect(t5?.metadata["quantization"] == "int4")

    let vae = registry.descriptor(for: "sdxl-vae-decoder-fp16")
    #expect(vae?.metadata["component_role"] == "decoder")
    #expect(vae?.metadata["quantization"] == "fp16")
  }

  @Test("Component IDs are accessible as constants")
  func componentIdConstants() {
    #expect(CatalogRegistration.t5XXLEncoderComponentId == "t5-xxl-encoder-int4")
    #expect(CatalogRegistration.sdxlVAEDecoderComponentId == "sdxl-vae-decoder-fp16")
  }
}
