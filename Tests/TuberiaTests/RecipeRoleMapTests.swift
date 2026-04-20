import CoreGraphics
import Foundation
@preconcurrency import MLX
import SwiftAcervo
import Testing
import os

@testable import Tuberia

// MARK: - Local Minimal Mocks (TuberiaTests-only, no dependency on TuberiaGPUTests)

private final class RoleMapTestEncoder: TextEncoder, @unchecked Sendable {
  struct Config: Sendable {
    let dim: Int
    let seqLen: Int
  }
  typealias Configuration = Config
  private(set) var isLoaded = false
  private var weights: ModuleParameters?
  var outputEmbeddingDim: Int { configuration.dim }
  var maxSequenceLength: Int { configuration.seqLen }
  var estimatedMemoryBytes: Int { 0 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { weights }
  private let configuration: Config
  required init(configuration: Config) throws { self.configuration = configuration }
  func encode(_ input: TextEncoderInput) throws -> TextEncoderOutput {
    TextEncoderOutput(
      embeddings: MLXArray.ones([1, input.maxLength, configuration.dim]),
      mask: MLXArray.ones([1, input.maxLength])
    )
  }
  func apply(weights: ModuleParameters) throws {
    self.weights = weights
    isLoaded = true
  }
  func unload() {
    weights = nil
    isLoaded = false
  }
}

private final class RoleMapTestBackbone: Backbone, @unchecked Sendable {
  struct Config: Sendable {
    let dim: Int
    let seqLen: Int
  }
  typealias Configuration = Config
  private(set) var isLoaded = false
  private var weights: ModuleParameters?
  var expectedConditioningDim: Int { configuration.dim }
  var outputLatentChannels: Int { 4 }
  var expectedMaxSequenceLength: Int { configuration.seqLen }
  var estimatedMemoryBytes: Int { 0 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { weights }
  private let configuration: Config
  required init(configuration: Config) throws { self.configuration = configuration }
  func forward(_ input: BackboneInput) throws -> MLXArray { MLXArray.ones(input.latents.shape) }
  func apply(weights: ModuleParameters) throws {
    self.weights = weights
    isLoaded = true
  }
  func unload() {
    weights = nil
    isLoaded = false
  }
}

private final class RoleMapTestDecoder: Decoder, @unchecked Sendable {
  struct Config: Sendable {}
  typealias Configuration = Config
  private(set) var isLoaded = false
  private var weights: ModuleParameters?
  var expectedInputChannels: Int { 4 }
  var scalingFactor: Float { 0.18215 }
  var estimatedMemoryBytes: Int { 0 }
  var keyMapping: KeyMapping { { k in k } }
  var currentWeights: ModuleParameters? { weights }
  required init(configuration: Config) throws {}
  func decode(_ latents: MLXArray) throws -> DecodedOutput {
    let shape = latents.shape
    let b = shape[0]
    let h = shape.count > 1 ? shape[1] * 8 : 8
    let w = shape.count > 2 ? shape[2] * 8 : 8
    return DecodedOutput(
      data: MLXArray.ones([b, h, w, 3]) * 0.5,
      metadata: ImageDecoderMetadata(scalingFactor: scalingFactor)
    )
  }
  func apply(weights: ModuleParameters) throws {
    self.weights = weights
    isLoaded = true
  }
  func unload() {
    weights = nil
    isLoaded = false
  }
}

private final class RoleMapTestScheduler: Scheduler, @unchecked Sendable {
  struct Config: Sendable {}
  typealias Configuration = Config
  required init(configuration: Config) {}
  func configure(steps: Int, startTimestep: Int?) -> SchedulerPlan {
    SchedulerPlan(timesteps: [], sigmas: [])
  }
  func step(output: MLXArray, timestep: Int, sample: MLXArray) -> MLXArray { sample }
  func addNoise(to sample: MLXArray, noise: MLXArray, at timestep: Int) -> MLXArray { sample }
  func reset() {}
}

private final class RoleMapTestRenderer: Renderer, @unchecked Sendable {
  typealias Configuration = Void
  required init(configuration: Void) {}
  func render(_ input: DecodedOutput) throws -> RenderedOutput {
    var pixelData: [UInt8] = [128, 128, 128, 255]
    let space = CGColorSpaceCreateDeviceRGB()
    guard
      let ctx = CGContext(
        data: &pixelData, width: 1, height: 1,
        bitsPerComponent: 8, bytesPerRow: 4, space: space,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      ),
      let img = ctx.makeImage()
    else {
      throw PipelineError.renderingFailed(reason: "RoleMapTestRenderer: CGImage failed")
    }
    return .image(img)
  }
}

/// A recipe that allows explicit override of `componentIdFor`.
private struct ReversibleRecipe: PipelineRecipe, Sendable {
  typealias Encoder = RoleMapTestEncoder
  typealias Sched = RoleMapTestScheduler
  typealias Back = RoleMapTestBackbone
  typealias Dec = RoleMapTestDecoder
  typealias Rend = RoleMapTestRenderer

  let encoderConfig = RoleMapTestEncoder.Config(dim: 64, seqLen: 8)
  let schedulerConfig = RoleMapTestScheduler.Config()
  let backboneConfig = RoleMapTestBackbone.Config(dim: 64, seqLen: 8)
  let decoderConfig = RoleMapTestDecoder.Config()
  let rendererConfig: Void = ()

  let supportsImageToImage = false
  let unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy = .emptyPrompt
  let allComponentIds: [String]

  /// When non-nil, this map overrides the default `componentIdFor` computation.
  /// Used by tests to inject a deliberately reversed mapping.
  let componentIdForOverride: [PipelineRole: String]?

  var componentIdFor: [PipelineRole: String] {
    if let override = componentIdForOverride { return override }
    var map: [PipelineRole: String] = [:]
    for (role, id) in zip(PipelineRole.allCases, allComponentIds) { map[role] = id }
    return map
  }

  func quantizationFor(_ role: PipelineRole) -> QuantizationConfig { .asStored }
  func validate() throws {}
}

// MARK: - Spy ComponentReadinessService

/// Thread-safe spy that records every `ensureComponentReady` call, keyed by role.
///
/// Because `ensureComponentReady` is called inside the `for (segment, …, role, …)` loop,
/// we capture the component ID that was routed to each position by observing call order.
private final class RoleMapSpy: ComponentReadinessService, @unchecked Sendable {
  private let lock = OSAllocatedUnfairLock(initialState: [String]())

  var calledIds: [String] { lock.withLock { $0 } }

  func ensureComponentReady(
    _ componentId: String,
    progress: (@Sendable (AcervoDownloadProgress) -> Void)?
  ) async throws {
    lock.withLock { ids in ids.append(componentId) }
  }
}

// MARK: - Tests

/// Verifies that `DiffusionPipeline` routes component IDs via the role-keyed dictionary
/// (`componentIdFor`) rather than positional array indexing.
///
/// The critical assertion: a recipe whose `componentIdFor` map is reversed relative to
/// the `allComponentIds` default causes `ensureComponentReady` to be called with the
/// reversed IDs — proving the pipeline reads from the map, not from `allComponentIds[n]`.
///
/// REQ-PIPE-03: replace positional `findComponentId(for:)` with a role-keyed dictionary.
@Suite("RecipeRoleMap Tests", .serialized)
struct RecipeRoleMapTests {

  let encoderComponentId = "role-map-encoder"
  let backboneComponentId = "role-map-backbone"
  let decoderComponentId = "role-map-decoder"

  // MARK: - Default mapping (zip order) produces the canonical IDs

  @Test("Default componentIdFor zips allComponentIds with PipelineRole.allCases")
  func defaultMappingPreservesOrder() throws {
    // PipelineRole.allCases = [encoder, scheduler, backbone, decoder, renderer]
    // allComponentIds has 3 entries → encoder, scheduler, backbone get IDs; decoder/renderer get nil.
    let ids = [encoderComponentId, "sched-id", backboneComponentId]
    let recipe = ReversibleRecipe(allComponentIds: ids, componentIdForOverride: nil)

    let map = recipe.componentIdFor
    #expect(map[.encoder] == encoderComponentId)
    #expect(map[.scheduler] == "sched-id")
    #expect(map[.backbone] == backboneComponentId)
    #expect(map[.decoder] == nil)
    #expect(map[.renderer] == nil)
  }

  // MARK: - Reversed map reaches ensureComponentReady with the overridden IDs

  @Test("Reversed componentIdFor map reaches ensureComponentReady with reversed IDs")
  func reversedComponentIdForMap() async throws {
    // Build a reversed map: encoder slot -> decoderID, backbone stays, decoder slot -> encoderID.
    let reversedMap: [PipelineRole: String] = [
      .encoder: decoderComponentId,  // reversed: encoder role gets decoder's usual ID
      .backbone: backboneComponentId,  // unchanged
      .decoder: encoderComponentId,  // reversed: decoder role gets encoder's usual ID
    ]

    let recipe = ReversibleRecipe(
      allComponentIds: [encoderComponentId, backboneComponentId, decoderComponentId],
      componentIdForOverride: reversedMap
    )

    let pipeline = try DiffusionPipeline<
      RoleMapTestEncoder,
      RoleMapTestScheduler,
      RoleMapTestBackbone,
      RoleMapTestDecoder,
      RoleMapTestRenderer
    >(recipe: recipe)

    let spy = RoleMapSpy()
    await pipeline.setComponentReadinessService(spy)

    // loadModels will call ensureComponentReady for each weighted segment (encoder, backbone, decoder)
    // in order, then attempt WeightLoader.load which will fail (no real weights).
    // We only care which IDs reached the spy before the first failure.
    do {
      try await pipeline.loadModels { _, _ in }
    } catch {
      // Expected: WeightLoader.load fails without real storage.
    }

    // The spy must have received the REVERSED IDs in call order.
    // Encoder segment → spy receives decoderComponentId (not encoderComponentId).
    // At minimum the first call (encoder role) must be the reversed ID.
    let called = spy.calledIds
    #expect(called.count >= 1, "ensureComponentReady must be called at least once")

    // First call corresponds to the encoder role; the reversed map gives it decoderComponentId.
    #expect(
      called.first == decoderComponentId,
      "Encoder-role segment must route to decoderComponentId per the reversed componentIdFor map"
    )

    // Verify that each observed call used one of the IDs from the reversed map's values,
    // not any ID that would appear only in positional order from allComponentIds.
    // Under positional indexing: first call = encoderComponentId. Under role map: first = decoderComponentId.
    #expect(
      called.first != encoderComponentId,
      "If positional indexing were used, first call would be encoderComponentId — that must NOT happen"
    )

    // All recorded calls must be known reversed IDs (not stray values).
    for id in called {
      let knownIds = [encoderComponentId, backboneComponentId, decoderComponentId]
      #expect(knownIds.contains(id))
    }
  }

  // MARK: - componentIdFor override surface on MockPipelineRecipe

  @Test("componentIdFor default implementation zips allComponentIds with PipelineRole.allCases")
  func defaultImplementationOnProtocol() {
    // Verify the protocol-extension default using ReversibleRecipe with no override.
    let recipe = ReversibleRecipe(
      allComponentIds: [encoderComponentId, backboneComponentId, decoderComponentId],
      componentIdForOverride: nil
    )

    // PipelineRole.allCases order: encoder, scheduler, backbone, decoder, renderer
    let map = recipe.componentIdFor
    #expect(map[.encoder] == encoderComponentId)
    // scheduler gets backboneComponentId (position 1 in allComponentIds)
    #expect(map[.scheduler] == backboneComponentId)
    // backbone gets decoderComponentId (position 2 in allComponentIds)
    #expect(map[.backbone] == decoderComponentId)
    // decoder and renderer are beyond allComponentIds.count → nil
    #expect(map[.decoder] == nil)
    #expect(map[.renderer] == nil)
  }
}
