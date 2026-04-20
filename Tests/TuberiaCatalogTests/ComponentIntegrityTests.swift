// ComponentIntegrityTests.swift
// TuberiaCatalogTests
//
// S6 REQ-INT-01 — Integrity failure integration test.
//
// Stages a valid synthetic safetensors file in a temp directory, registers a
// ComponentDescriptor whose SHA-256 matches the file, then corrupts a single byte
// before calling AcervoManager.shared.withComponentAccess(_:in:). Asserts that
// AcervoError.integrityCheckFailed is thrown.
//
// Uses the internal withComponentAccess(_:in:perform:) overload (accessible via
// @testable import SwiftAcervo) to inject a temp base directory, avoiding any
// mutation of the global Acervo.customBaseDirectory.
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

// MARK: - ComponentIntegrityTests

@Suite("Component Integrity Tests", .serialized)
struct ComponentIntegrityTests {

  // MARK: - Integrity Failure

  /// Stages a synthetic safetensors file, registers a ComponentDescriptor with the
  /// correct SHA-256, then corrupts a single byte in the file. Asserts that
  /// AcervoManager.shared.withComponentAccess(_:in:) throws
  /// AcervoError.integrityCheckFailed.
  ///
  /// This directly verifies that Acervo's per-file SHA-256 check fires before
  /// the component's contents are exposed to the caller.
  @Test("testIntegrityFailure: corrupted file causes withComponentAccess to throw integrityCheckFailed")
  func testIntegrityFailure() async throws {
    // 1. Create a temp base directory for this test only.
    let tempBase = FileManager.default.temporaryDirectory
      .appendingPathComponent("TuberiaS6-integrity-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempBase, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tempBase) }

    let uid = UUID().uuidString.prefix(8)
    let componentId = "test-integrity-\(uid)"
    let repoId = "test-tuberia/integrity-\(uid)"
    let slug = repoId.replacingOccurrences(of: "/", with: "_")

    // Create: tempBase / slug / "model.safetensors"
    let componentDir = tempBase.appendingPathComponent(slug)
    try FileManager.default.createDirectory(at: componentDir, withIntermediateDirectories: true)
    let safetensorsURL = componentDir.appendingPathComponent("model.safetensors")

    // 2. Write a valid synthetic safetensors file.
    let tensor = MLXArray([Float(1.0)])
    try MLX.save(arrays: ["weight": tensor], url: safetensorsURL)

    // 3. Record the SHA-256 of the CORRECT content.
    let correctData = try Data(contentsOf: safetensorsURL)
    let correctSHA256 = CryptoKit.SHA256.hash(data: correctData)
      .map { String(format: "%02x", $0) }.joined()

    // 4. Register the descriptor with the correct hash.
    let descriptor = ComponentDescriptor(
      id: componentId,
      type: .backbone,
      displayName: "Test Integrity \(uid)",
      repoId: repoId,
      files: [
        ComponentFile(
          relativePath: "model.safetensors",
          expectedSizeBytes: Int64(correctData.count),
          sha256: correctSHA256
        )
      ],
      estimatedSizeBytes: Int64(correctData.count),
      minimumMemoryBytes: 1024
    )
    Acervo.register(descriptor)
    defer { Acervo.unregister(componentId) }

    // 5. Corrupt a single byte — XOR the last byte so the hash no longer matches.
    var corruptData = correctData
    let lastIndex = corruptData.index(before: corruptData.endIndex)
    corruptData[lastIndex] = corruptData[lastIndex] ^ 0xFF
    try corruptData.write(to: safetensorsURL)

    // Sanity: corrupted SHA-256 must differ.
    let corruptedSHA256 = CryptoKit.SHA256.hash(data: corruptData)
      .map { String(format: "%02x", $0) }.joined()
    #expect(
      corruptedSHA256 != correctSHA256,
      "Pre-condition: corrupted file must differ in SHA-256"
    )

    // 6. Call withComponentAccess(_:in:) and assert the integrity failure fires.
    do {
      _ = try await AcervoManager.shared.withComponentAccess(
        componentId,
        in: tempBase
      ) { handle -> String in
        return "should not reach closure — integrity check must fire first"
      }
      Issue.record("Expected integrityCheckFailed but withComponentAccess succeeded")
    } catch let acervoError as AcervoError {
      guard case .integrityCheckFailed(let file, let expected, let actual) = acervoError else {
        Issue.record("Expected .integrityCheckFailed, got: \(acervoError)")
        return
      }
      // The file path must name the safetensors file.
      #expect(file.hasSuffix("model.safetensors") || file == "model.safetensors")
      // expected must equal the registered hash.
      #expect(expected == correctSHA256)
      // actual must equal the corrupted hash.
      #expect(actual == corruptedSHA256)
    } catch {
      Issue.record("Expected AcervoError.integrityCheckFailed, got: \(error)")
    }
  }
}
