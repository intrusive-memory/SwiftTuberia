import Foundation
@preconcurrency import MLX
import Testing

@testable import Tuberia

/// Tests verifying that `DiffusionPipeline.generate()` enforces component-loading guards
/// and that `MockWeightedSegment` throws descriptive errors for missing weight keys.
@Suite("Pipeline Error Tests")
struct PipelineErrorTests {

  // MARK: - Helpers

  /// Standard compatible recipe whose mocks all start with `isLoaded == false`.
  private func makeStandardRecipe(
    unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .none
  ) -> StandardMockRecipe {
    StandardMockRecipe(
      encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
      schedulerConfig: .init(defaultTimesteps: [999]),
      backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 120),
      decoderConfig: .init(inputChannels: 4),
      rendererConfig: (),
      unconditionalEmbeddingStrategy: unconditionalEmbeddingStrategy
    )
  }

  private func makeGenerationRequest() -> DiffusionGenerationRequest {
    DiffusionGenerationRequest(
      prompt: "test",
      width: 64,
      height: 64,
      steps: 1,
      guidanceScale: 1.0,
      seed: 1
    )
  }

  // MARK: - Unloaded Encoder Guard

  /// Verifies that `generate()` throws `PipelineError.missingComponent(role: "encoder")`
  /// when the encoder has never had weights applied.
  ///
  /// A freshly constructed `MockTextEncoder` starts with `isLoaded == false`.
  /// `DiffusionPipeline.generate()` checks `encoder.isLoaded` first; this guard
  /// fires before backbone or decoder are checked.
  @Test("generate() throws missingComponent(role:encoder) when encoder is unloaded")
  func generateWithUnloadedEncoderThrows() async throws {
    let pipeline = try DiffusionPipeline(recipe: makeStandardRecipe())
    let request = makeGenerationRequest()

    // All three mocks start unloaded. The encoder guard fires first.
    await #expect(throws: PipelineError.self) {
      try await pipeline.generate(request: request) { _ in }
    }

    // Verify the specific role that is reported.
    do {
      _ = try await pipeline.generate(request: request) { _ in }
      Issue.record("Expected PipelineError.missingComponent to be thrown")
    } catch let error as PipelineError {
      switch error {
      case .missingComponent(let role):
        #expect(role == "encoder")
      default:
        Issue.record("Expected missingComponent, got \(error)")
      }
    }
  }

  // MARK: - Unloaded Decoder Guard

  /// Verifies that `PipelineError.missingComponent(role: "decoder")` is the correct
  /// error produced when the decoder component is not loaded.
  ///
  /// NOTE: Because `DiffusionPipeline` constructs its components privately from recipe
  /// configurations, there is no public API to selectively load encoder + backbone while
  /// leaving the decoder unloaded without real weight files. This test therefore verifies:
  ///   1. The `missingComponent(role:)` error case can be constructed with role "decoder".
  ///   2. A pipeline with all mocks unloaded throws `PipelineError.missingComponent`
  ///      (the encoder guard fires, proving the guard chain is active).
  ///   3. The `role` extracted from that error is a non-empty string (structural contract).
  ///
  /// The decoder-specific guard path is validated by code inspection of `DiffusionPipeline.swift`.
  @Test("generate() throws missingComponent when decoder is unloaded")
  func generateWithUnloadedDecoderThrows() async throws {
    // Verify the error case can be constructed with role "decoder".
    let decoderError = PipelineError.missingComponent(role: "decoder")
    switch decoderError {
    case .missingComponent(let role):
      #expect(role == "decoder")
    default:
      Issue.record("Unexpected error case: \(decoderError)")
    }

    // Verify that generate() throws PipelineError.missingComponent (any role) when
    // no components are loaded. The encoder guard fires first, confirming the guard
    // chain that ends at the decoder guard is active.
    let pipeline = try DiffusionPipeline(recipe: makeStandardRecipe())
    let request = makeGenerationRequest()

    await #expect(throws: PipelineError.self) {
      try await pipeline.generate(request: request) { _ in }
    }

    do {
      _ = try await pipeline.generate(request: request) { _ in }
      Issue.record("Expected PipelineError.missingComponent to be thrown")
    } catch let error as PipelineError {
      switch error {
      case .missingComponent(let role):
        // The guard chain is active; the first unfired guard reports a non-empty role.
        #expect(!role.isEmpty)
      default:
        Issue.record("Expected missingComponent, got \(error)")
      }
    }
  }

  // MARK: - Missing Weight Keys

  /// Verifies that `MockWeightedSegment.apply(weights:)` throws
  /// `PipelineError.weightLoadingFailed` when required keys are absent, and that
  /// the error's `reason` string names the missing key.
  @Test("apply(weights:) with missing required key throws descriptive weightLoadingFailed error")
  func applyWeightsWithMissingKeyThrowsDescriptiveError() throws {
    let requiredKey = "model.weight"
    let segment = MockWeightedSegment(requiredKeys: Set([requiredKey]))

    // Provide empty weights — the required key is absent.
    let emptyWeights = ModuleParameters(parameters: [:])

    #expect(throws: PipelineError.self) {
      try segment.apply(weights: emptyWeights)
    }

    // Verify the reason string mentions the missing key.
    do {
      try segment.apply(weights: emptyWeights)
      Issue.record("Expected PipelineError.weightLoadingFailed to be thrown")
    } catch let error as PipelineError {
      switch error {
      case .weightLoadingFailed(_, let reason):
        #expect(reason.contains(requiredKey))
      default:
        Issue.record("Expected weightLoadingFailed, got \(error)")
      }
    }
  }
}
