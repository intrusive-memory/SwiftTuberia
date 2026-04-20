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

// MARK: - ComponentFile Integrity Assertions (S2 REQ-T4)

/// Validates that every registered ComponentFile carries a non-nil SHA-256 digest
/// and a positive expectedSizeBytes value, and that the digest is exactly 64
/// lowercase hex characters.
///
/// These assertions guard against accidental removal of checksum data and ensure
/// the integrity-verification path in SwiftAcervo can validate every file.
@Suite("ComponentFile Integrity Assertions", .serialized)
struct ComponentFileIntegrityTests {

  /// Returns true when `hex` is exactly 64 lowercase hex characters (SHA-256 format).
  private func isValidSHA256(_ hex: String) -> Bool {
    guard hex.count == 64 else { return false }
    return hex.allSatisfy { $0.isHexDigit && ($0.isNumber || $0.isLowercase) }
  }

  /// Collect all ComponentFile entries from all registered descriptors.
  private func allRegisteredFiles() -> [(componentId: String, file: ComponentFile)] {
    let registry = CatalogRegistration.shared
    registry.ensureRegistered()
    var results: [(componentId: String, file: ComponentFile)] = []
    for id in [
      CatalogRegistration.t5XXLEncoderComponentId,
      CatalogRegistration.sdxlVAEDecoderComponentId,
    ] {
      if let descriptor = registry.descriptor(for: id) {
        for file in descriptor.files {
          results.append((componentId: id, file: file))
        }
      }
    }
    return results
  }

  @Test("Every ComponentFile has a non-nil sha256")
  func allFilesHaveSHA256() {
    let files = allRegisteredFiles()
    #expect(files.isEmpty == false, "Expected at least one registered ComponentFile")
    for entry in files {
      #expect(
        entry.file.sha256 != nil,
        "ComponentFile '\(entry.file.relativePath)' in '\(entry.componentId)' is missing sha256"
      )
    }
  }

  @Test("Every ComponentFile has expectedSizeBytes > 0")
  func allFilesHavePositiveSize() {
    let files = allRegisteredFiles()
    #expect(files.isEmpty == false, "Expected at least one registered ComponentFile")
    for entry in files {
      let size = entry.file.expectedSizeBytes
      #expect(
        size != nil,
        "ComponentFile '\(entry.file.relativePath)' in '\(entry.componentId)' is missing expectedSizeBytes"
      )
      if let size {
        #expect(
          size > 0,
          "ComponentFile '\(entry.file.relativePath)' in '\(entry.componentId)' has non-positive expectedSizeBytes: \(size)"
        )
      }
    }
  }

  @Test("Every ComponentFile sha256 matches ^[0-9a-f]{64}$")
  func allFilesHaveValidSHA256Format() {
    let files = allRegisteredFiles()
    #expect(files.isEmpty == false, "Expected at least one registered ComponentFile")
    for entry in files {
      guard let sha256 = entry.file.sha256 else {
        Issue.record("ComponentFile '\(entry.file.relativePath)' in '\(entry.componentId)' is missing sha256")
        continue
      }
      #expect(
        isValidSHA256(sha256),
        "ComponentFile '\(entry.file.relativePath)' in '\(entry.componentId)' has invalid sha256: '\(sha256)'"
      )
    }
  }

  @Test("T5-XXL encoder files all have populated checksums")
  func t5XXLFilesAllPopulated() {
    let registry = CatalogRegistration.shared
    registry.ensureRegistered()
    let descriptor = registry.descriptor(for: CatalogRegistration.t5XXLEncoderComponentId)
    #expect(descriptor != nil)
    guard let descriptor else { return }
    for file in descriptor.files {
      #expect(file.sha256 != nil, "T5 file '\(file.relativePath)' missing sha256")
      #expect((file.expectedSizeBytes ?? 0) > 0, "T5 file '\(file.relativePath)' has no expectedSizeBytes")
    }
  }

  @Test("SDXL VAE decoder files all have populated checksums")
  func sdxlVAEFilesAllPopulated() {
    let registry = CatalogRegistration.shared
    registry.ensureRegistered()
    let descriptor = registry.descriptor(for: CatalogRegistration.sdxlVAEDecoderComponentId)
    #expect(descriptor != nil)
    guard let descriptor else { return }
    for file in descriptor.files {
      #expect(file.sha256 != nil, "VAE file '\(file.relativePath)' missing sha256")
      #expect((file.expectedSizeBytes ?? 0) > 0, "VAE file '\(file.relativePath)' has no expectedSizeBytes")
    }
  }
}
