import Foundation
@preconcurrency import MLX
import Testing

@testable import Tuberia

/// Tests for REQ-MEM-01: the text encoder is freed after the encode phase of a
/// generation (before the denoise loop) and transparently reloaded on the next
/// generation. This is the change that keeps PixArt's ~1.2 GB int4 T5-XXL encoder
/// out of the memory-heavy denoise/decode phases so it stops OOM/jetsam-killing
/// on iOS.
@Suite("Phased Encoder Memory (REQ-MEM-01)")
struct PhasedEncoderMemoryTests {

  private func loadedPipeline() async throws -> DiffusionPipeline<
    MockTextEncoder, MockScheduler, MockBackbone, MockDecoder, MockRenderer
  > {
    try await StandardMockRecipe.loaded(
      encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
      schedulerConfig: .init(defaultTimesteps: [999, 500, 0]),
      backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 120),
      decoderConfig: .init(inputChannels: 4),
      rendererConfig: (),
      unconditionalEmbeddingStrategy: .none
    )
  }

  private func request() -> DiffusionGenerationRequest {
    DiffusionGenerationRequest(
      prompt: "a red car on a cobblestone street",
      width: 64,
      height: 64,
      steps: 3,
      guidanceScale: 1.0,
      seed: 42
    )
  }

  @Test("freeEncoderAfterEncode frees the encoder during generation")
  func phasedUnloadFreesEncoder() async throws {
    let pipeline = try await loadedPipeline()
    await pipeline.setFreeEncoderAfterEncode(true)

    let encoder = await pipeline.encoder
    #expect(encoder.isLoaded, "encoder should be loaded before generation")

    let result = try await pipeline.generate(request: request()) { _ in }

    // Generation still completes...
    switch result.output {
    case .image: break
    default: Issue.record("Expected .image output, got \(result.output)")
    }
    // ...but the encoder is no longer resident (freed after encode, before denoise).
    #expect(!encoder.isLoaded, "encoder should be unloaded after the encode phase")
    #expect(encoder.encodeCallCount == 1, "encoder should have encoded exactly once")
  }

  @Test("default (macOS) keeps the encoder resident across a generation")
  func defaultKeepsEncoderResident() async throws {
    let pipeline = try await loadedPipeline()
    // Do NOT enable freeEncoderAfterEncode — exercise the macOS default (false).
    await pipeline.setFreeEncoderAfterEncode(false)

    let encoder = await pipeline.encoder
    _ = try await pipeline.generate(request: request()) { _ in }

    #expect(encoder.isLoaded, "encoder must stay resident when phasing is disabled")
  }

  @Test("second generation transparently reloads the freed encoder")
  func secondGenerationReloadsEncoder() async throws {
    let pipeline = try await loadedPipeline()
    await pipeline.setFreeEncoderAfterEncode(true)

    let encoder = await pipeline.encoder
    // Reload seam: mocks re-apply synthetic weights without Acervo/disk access,
    // standing in for the production WeightLoader reload path.
    await pipeline.setEncoderReloadOverride {
      try encoder.apply(weights: ModuleParameters(parameters: ["w": MLXArray.ones([2, 2])]))
    }

    // Generation #1 — encoder gets freed after encode.
    _ = try await pipeline.generate(request: request()) { _ in }
    #expect(!encoder.isLoaded, "encoder freed after generation #1")

    // Generation #2 — load-once/generate-many: the encoder must be restored and
    // used again, not left unloaded (which would throw missingComponent or, worse,
    // silently produce blank conditioning).
    let result2 = try await pipeline.generate(request: request()) { _ in }
    switch result2.output {
    case .image: break
    default: Issue.record("Expected .image output on generation #2, got \(result2.output)")
    }
    #expect(encoder.encodeCallCount == 2, "encoder should have encoded on both generations")
    #expect(!encoder.isLoaded, "encoder freed again after generation #2")
  }
}
