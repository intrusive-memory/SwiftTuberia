import Foundation
import Testing
import Tuberia

@testable import TuberiaCatalog

@Suite("CatalogRegistration Tests", .serialized)
struct CatalogRegistrationTests {

  @Test("ensureRegistered() registers expected component IDs")
  func registersExpectedComponents() {
    let registry = CatalogRegistration.shared
    registry.reset()  // start clean for test

    registry.ensureRegistered()

    let ids = registry.registeredComponentIds()
    #expect(ids.contains("t5-xxl-encoder-int4"))
    #expect(ids.contains("sdxl-vae-decoder-fp16"))
  }

  @Test("Registration is idempotent -- duplicate registration does not error")
  func idempotentRegistration() {
    let registry = CatalogRegistration.shared
    registry.reset()

    // Register twice -- should not throw or duplicate
    registry.ensureRegistered()
    registry.ensureRegistered()

    let ids = registry.registeredComponentIds()
    #expect(ids.count == 2)
  }

  @Test("T5-XXL descriptor has correct HuggingFace repo")
  func t5XXLDescriptor() {
    let descriptor = CatalogRegistration.t5XXLEncoderDescriptor
    #expect(descriptor.componentId == "t5-xxl-encoder-int4")
    #expect(descriptor.componentType == .encoder)
    #expect(descriptor.huggingFaceRepo == "intrusive-memory/t5-xxl-int4-mlx")
    #expect(descriptor.estimatedSizeBytes > 1_000_000_000)  // > 1 GB
  }

  @Test("SDXL VAE descriptor has correct HuggingFace repo")
  func sdxlVAEDescriptor() {
    let descriptor = CatalogRegistration.sdxlVAEDecoderDescriptor
    #expect(descriptor.componentId == "sdxl-vae-decoder-fp16")
    #expect(descriptor.componentType == .decoder)
    #expect(descriptor.huggingFaceRepo == "intrusive-memory/sdxl-vae-fp16-mlx")
    #expect(descriptor.estimatedSizeBytes > 100_000_000)  // > 100 MB
  }

  @Test("descriptor(for:) returns correct descriptor by ID")
  func lookupById() {
    let registry = CatalogRegistration.shared
    registry.reset()
    registry.ensureRegistered()

    let t5 = registry.descriptor(for: "t5-xxl-encoder-int4")
    #expect(t5 != nil)
    #expect(t5?.huggingFaceRepo == "intrusive-memory/t5-xxl-int4-mlx")

    let vae = registry.descriptor(for: "sdxl-vae-decoder-fp16")
    #expect(vae != nil)
    #expect(vae?.huggingFaceRepo == "intrusive-memory/sdxl-vae-fp16-mlx")
  }

  @Test("descriptor(for:) returns nil for unknown ID")
  func unknownIdReturnsNil() {
    let registry = CatalogRegistration.shared
    registry.reset()
    registry.ensureRegistered()

    let unknown = registry.descriptor(for: "nonexistent-component")
    #expect(unknown == nil)
  }

  @Test("isComponentRegistered returns true for known components")
  func isRegisteredCheck() {
    let registry = CatalogRegistration.shared
    registry.reset()
    registry.ensureRegistered()

    #expect(registry.isComponentRegistered("t5-xxl-encoder-int4"))
    #expect(registry.isComponentRegistered("sdxl-vae-decoder-fp16"))
    #expect(!registry.isComponentRegistered("unknown-component"))
  }

  @Test("huggingFaceRepo returns repo for known components")
  func huggingFaceRepoLookup() {
    let registry = CatalogRegistration.shared
    registry.reset()
    registry.ensureRegistered()

    #expect(
      registry.huggingFaceRepo(for: "t5-xxl-encoder-int4") == "intrusive-memory/t5-xxl-int4-mlx")
    #expect(
      registry.huggingFaceRepo(for: "sdxl-vae-decoder-fp16") == "intrusive-memory/sdxl-vae-fp16-mlx"
    )
    #expect(registry.huggingFaceRepo(for: "unknown") == nil)
  }

  @Test("Manual registration of duplicate component is silently ignored")
  func duplicateRegistrationIgnored() {
    let registry = CatalogRegistration.shared
    registry.reset()

    let descriptor = ComponentDescriptor(
      componentId: "test-component",
      componentType: .backbone,
      huggingFaceRepo: "test/repo",
      filePatterns: ["*.safetensors"],
      estimatedSizeBytes: 1000
    )

    registry.register(descriptor)
    registry.register(descriptor)  // duplicate

    let ids = registry.registeredComponentIds()
    let testCount = ids.filter { $0 == "test-component" }.count
    #expect(testCount == 1, "Should have exactly one entry for the component")
  }

  @Test("Descriptors include expected file patterns")
  func filePatterns() {
    let t5 = CatalogRegistration.t5XXLEncoderDescriptor
    #expect(t5.filePatterns.contains("*.safetensors"))
    #expect(t5.filePatterns.contains("tokenizer.json"))
    #expect(t5.filePatterns.contains("config.json"))

    let vae = CatalogRegistration.sdxlVAEDecoderDescriptor
    #expect(vae.filePatterns.contains("*.safetensors"))
    #expect(vae.filePatterns.contains("config.json"))
  }

  @Test("SHA-256 checksums are nil initially (backfilled after weight conversion)")
  func checksumNil() {
    let t5 = CatalogRegistration.t5XXLEncoderDescriptor
    #expect(t5.sha256Checksums == nil)

    let vae = CatalogRegistration.sdxlVAEDecoderDescriptor
    #expect(vae.sha256Checksums == nil)
  }
}
