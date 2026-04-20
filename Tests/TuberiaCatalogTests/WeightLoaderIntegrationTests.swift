// WeightLoaderIntegrationTests.swift
// TuberiaCatalogTests
//
// S6 REQ-INT-01 — End-to-end integration tests for WeightLoader via
// AcervoManager.withComponentAccess(_:in:perform:).
//
// Tests use a temporary directory as the base for all Acervo operations,
// injected via the internal withComponentAccess(_:in:perform:) overload.
// This avoids mutating the global Acervo.customBaseDirectory and is safe
// for concurrent test execution.
//
// Synthetic files are created with MLX.save(arrays:url:) — the same format
// WeightLoader reads. Each file is < 1 MB (single float32 scalar per key).
//
// Test inventory:
//   testHappyPath              — loads weights via withComponentAccess, asserts key present
//   testNotDownloaded          — missing files → AcervoError.componentNotDownloaded
//   testLoRALocalAccess        — bare safetensors → LoRALoader.loadAdapterWeights
//
// No network, no URLSession, no CDN URLs, no timed waits, no global state mutation.

import CryptoKit
import Foundation
@preconcurrency import MLX
import Testing
import SwiftAcervo
import Tuberia

@testable import TuberiaCatalog
@testable import SwiftAcervo

// MARK: - Shared Helpers

/// Computes SHA-256 of a file using CryptoKit (mirrors AcervoError integrity check).
private func sha256Hex(of url: URL) throws -> String {
  let data = try Data(contentsOf: url)
  let digest = CryptoKit.SHA256.hash(data: data)
  return digest.map { String(format: "%02x", $0) }.joined()
}

/// Creates a unique temporary directory. Caller must clean up.
private func makeTempDir(label: String) throws -> URL {
  let dir = FileManager.default.temporaryDirectory
    .appendingPathComponent("TuberiaS6-\(label)-\(UUID().uuidString)")
  try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
  return dir
}

/// Removes a path silently.
private func silentRemove(_ url: URL) {
  try? FileManager.default.removeItem(at: url)
}

// MARK: - WeightLoader Integration Tests

@Suite("WeightLoader Integration Tests", .serialized)
struct WeightLoaderIntegrationTests {

  // MARK: - Happy Path

  /// Stages a synthetic safetensors file in a temp base directory, registers a
  /// ComponentDescriptor whose SHA-256 matches the file, then calls
  /// AcervoManager.shared.withComponentAccess(_:in:) to obtain a ComponentHandle
  /// and loads the safetensors using MLX.loadArrays(url:). Asserts the expected
  /// tensor key is present in the resulting ModuleParameters.
  ///
  /// This exercises the core path used by WeightLoader.load internally.
  @Test("testHappyPath: withComponentAccess provides handle; loadArrays returns expected key")
  func testHappyPath() async throws {
    let tempBase = try makeTempDir(label: "happy")
    defer { silentRemove(tempBase) }

    let uid = UUID().uuidString.prefix(8)
    let componentId = "test-happy-\(uid)"
    let repoId = "test-tuberia/happy-\(uid)"
    let slug = repoId.replacingOccurrences(of: "/", with: "_")
    let tensorKey = "layers.0.weight"

    // Create: tempBase / slug / "model.safetensors"
    let componentDir = tempBase.appendingPathComponent(slug)
    try FileManager.default.createDirectory(at: componentDir, withIntermediateDirectories: true)
    let safetensorsURL = componentDir.appendingPathComponent("model.safetensors")

    // Write a synthetic safetensors containing a single float32 scalar.
    let tensor = MLXArray([Float(1.0)])
    try MLX.save(arrays: [tensorKey: tensor], url: safetensorsURL)

    // Compute SHA-256 so the integrity check passes.
    let sha256 = try sha256Hex(of: safetensorsURL)
    let fileSize = Int64(try Data(contentsOf: safetensorsURL).count)

    // Register a test-only descriptor.
    let descriptor = ComponentDescriptor(
      id: componentId,
      type: .backbone,
      displayName: "Test Happy \(uid)",
      repoId: repoId,
      files: [
        ComponentFile(
          relativePath: "model.safetensors",
          expectedSizeBytes: fileSize,
          sha256: sha256
        )
      ],
      estimatedSizeBytes: fileSize,
      minimumMemoryBytes: 1024
    )
    Acervo.register(descriptor)
    defer { Acervo.unregister(componentId) }

    // Call the internal withComponentAccess(_:in:perform:) overload with our tempBase.
    // This is the same internal path WeightLoader.load uses, but with a custom base
    // directory rather than the real App Group container.
    let params = try await AcervoManager.shared.withComponentAccess(
      componentId,
      in: tempBase
    ) { handle -> ModuleParameters in
      let urls = try handle.urls(matching: ".safetensors")
      guard !urls.isEmpty else {
        throw PipelineError.weightLoadingFailed(
          component: componentId, reason: "No .safetensors in handle")
      }
      var parameters: [String: MLXArray] = [:]
      for url in urls {
        let arrays = try loadArrays(url: url)
        for (key, value) in arrays {
          parameters[key] = value
        }
      }
      return ModuleParameters(parameters: parameters)
    }

    #expect(
      params.parameters[tensorKey] != nil,
      "Expected '\(tensorKey)' in loaded ModuleParameters"
    )
    #expect(!params.parameters.isEmpty, "Expected non-empty ModuleParameters")
  }

  // MARK: - Not-Downloaded Failure

  /// Registers a descriptor but stages NO files on disk, then calls
  /// AcervoManager.shared.withComponentAccess(_:in:). Asserts it throws
  /// AcervoError.componentNotDownloaded (the condition WeightLoader maps to
  /// PipelineError.modelNotDownloaded).
  @Test("testNotDownloaded: withComponentAccess throws componentNotDownloaded when files absent")
  func testNotDownloaded() async throws {
    let tempBase = try makeTempDir(label: "notdl")
    defer { silentRemove(tempBase) }

    let uid = UUID().uuidString.prefix(8)
    let componentId = "test-not-dl-\(uid)"

    // Register descriptor — intentionally no files on disk.
    let descriptor = ComponentDescriptor(
      id: componentId,
      type: .backbone,
      displayName: "Test Not Downloaded \(uid)",
      repoId: "test-tuberia/not-dl-\(uid)",
      files: [
        ComponentFile(
          relativePath: "model.safetensors",
          expectedSizeBytes: nil,
          sha256: nil
        )
      ],
      estimatedSizeBytes: 1024,
      minimumMemoryBytes: 1024
    )
    Acervo.register(descriptor)
    defer { Acervo.unregister(componentId) }

    do {
      _ = try await AcervoManager.shared.withComponentAccess(
        componentId,
        in: tempBase
      ) { handle -> String in
        return "should not reach"
      }
      Issue.record("Expected componentNotDownloaded but withComponentAccess succeeded")
    } catch let acervoError as AcervoError {
      guard case .componentNotDownloaded(let id) = acervoError else {
        Issue.record("Expected .componentNotDownloaded, got: \(acervoError)")
        return
      }
      #expect(id == componentId)
    } catch {
      Issue.record("Expected AcervoError.componentNotDownloaded, got: \(error)")
    }
  }

  // MARK: - LoRA Local Access

  /// Stages a safetensors file with LoRA-shaped tensors in a temp directory.
  /// Calls LoRALoader.loadAdapterWeights(config: .init(localPath:)) and asserts
  /// the returned ModuleParameters is non-empty.
  ///
  /// Uses AcervoManager.withLocalAccess (no component registry, no SHA check).
  @Test("testLoRALocalAccess: LoRALoader.loadAdapterWeights loads local safetensors")
  func testLoRALocalAccess() async throws {
    let tempDir = try makeTempDir(label: "lora")
    defer { silentRemove(tempDir) }

    let safetensorsURL = tempDir.appendingPathComponent("adapter.safetensors")
    let loraA = MLXArray([Float(0.1), Float(0.2)])
    let loraB = MLXArray([Float(0.3), Float(0.4)])
    try MLX.save(
      arrays: [
        "layers.0.weight.lora_A": loraA,
        "layers.0.weight.lora_B": loraB,
      ],
      url: safetensorsURL
    )

    let config = LoRAConfig(localPath: safetensorsURL.path)
    let params = try await LoRALoader.loadAdapterWeights(
      config: config,
      keyMapping: { key in key }
    )

    #expect(!params.parameters.isEmpty, "Expected non-empty LoRA adapter ModuleParameters")
    #expect(
      params.parameters["layers.0.weight.lora_A"] != nil,
      "Expected lora_A key in adapter params"
    )
    #expect(
      params.parameters["layers.0.weight.lora_B"] != nil,
      "Expected lora_B key in adapter params"
    )
  }
}
