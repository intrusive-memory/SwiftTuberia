// WeightLoaderIntegrationTests.swift
// TuberiaCatalogTests
//
// Tests exercising Tuberia-owned loader code. Acervo-owned behavior (component
// access resolution, SHA-256 integrity verification, not-downloaded detection)
// is tested in SwiftAcervo's own test suite and must not be re-tested here.
//
// Test inventory:
//   testLoRALocalAccess  — LoRALoader.loadAdapterWeights over a local file
//
// No network, no URLSession, no CDN URLs, no timed waits.

import Foundation
@preconcurrency import MLX
import Testing
import SwiftAcervo
import Tuberia

@testable import TuberiaCatalog

/// Creates a unique temporary directory. Caller must clean up.
private func makeTempDir(label: String) throws -> URL {
  let dir = FileManager.default.temporaryDirectory
    .appendingPathComponent("TuberiaLoader-\(label)-\(UUID().uuidString)")
  try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
  return dir
}

/// Removes a path silently.
private func silentRemove(_ url: URL) {
  try? FileManager.default.removeItem(at: url)
}

@Suite("WeightLoader Integration Tests", .serialized)
struct WeightLoaderIntegrationTests {

  /// Stages a safetensors file with LoRA-shaped tensors and asserts that
  /// LoRALoader.loadAdapterWeights returns a populated ModuleParameters.
  /// Covers Tuberia's LoRA loader wrapping of AcervoManager.withLocalAccess.
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
