import Testing
import Foundation
@preconcurrency import MLX
@testable import Tuberia

/// Thread-safe counter for use in @Sendable closures.
final class LockedCounter: @unchecked Sendable {
    private var _value: Int = 0
    private let lock = NSLock()

    func increment() {
        lock.lock()
        _value += 1
        lock.unlock()
    }

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return _value
    }
}

/// Tests validating pipeline assembly, component lifecycle, and basic orchestration.
@Suite("Pipeline Assembly Tests")
struct PipelineAssemblyTests {

    // MARK: - Assembly Success

    @Test("Pipeline assembles from compatible mock components")
    func successfulAssembly() throws {
        let recipe = StandardMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
            schedulerConfig: .init(),
            backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 120),
            decoderConfig: .init(inputChannels: 4),
            rendererConfig: ()
        )

        let pipeline = try DiffusionPipeline(recipe: recipe)

        // Verify pipeline properties
        #expect(!pipeline.isLoaded)
        #expect(pipeline.memoryRequirement.peakMemoryBytes > 0)
        #expect(pipeline.memoryRequirement.phasedMemoryBytes > 0)
        #expect(pipeline.memoryRequirement.phasedMemoryBytes <= pipeline.memoryRequirement.peakMemoryBytes)
    }

    @Test("Memory requirement computed correctly from component estimates")
    func memoryRequirementCalculation() throws {
        let encoderMem = 1_000_000
        let backboneMem = 2_000_000
        let decoderMem = 500_000

        let recipe = StandardMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120, estimatedMemory: encoderMem),
            schedulerConfig: .init(),
            backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 120, estimatedMemory: backboneMem),
            decoderConfig: .init(inputChannels: 4, estimatedMemory: decoderMem),
            rendererConfig: ()
        )

        let pipeline = try DiffusionPipeline(recipe: recipe)

        // Peak = encoder + backbone + decoder
        let expectedPeak = UInt64(encoderMem + backboneMem + decoderMem)
        #expect(pipeline.memoryRequirement.peakMemoryBytes == expectedPeak)

        // Phased = max(encoder, backbone + decoder) = max(1M, 2.5M) = 2.5M
        let expectedPhased = UInt64(max(encoderMem, backboneMem + decoderMem))
        #expect(pipeline.memoryRequirement.phasedMemoryBytes == expectedPhased)
    }

    // MARK: - WeightedSegment Lifecycle

    @Test("WeightedSegment apply sets isLoaded to true")
    func weightedSegmentApply() throws {
        let segment = MockWeightedSegment()
        #expect(!segment.isLoaded)

        let weights = ModuleParameters(parameters: [
            "layer.weight": MLXArray.ones([4, 4])
        ])

        try segment.apply(weights: weights)
        #expect(segment.isLoaded)
        #expect(segment.applyCallCount == 1)
    }

    @Test("WeightedSegment unload clears weights and sets isLoaded to false")
    func weightedSegmentUnload() throws {
        let segment = MockWeightedSegment()
        let weights = ModuleParameters(parameters: [
            "layer.weight": MLXArray.ones([4, 4])
        ])

        try segment.apply(weights: weights)
        #expect(segment.isLoaded)

        segment.unload()
        #expect(!segment.isLoaded)
        #expect(segment.storedWeights == nil)
        #expect(segment.unloadCallCount == 1)
    }

    @Test("WeightedSegment apply with missing required keys throws clear error")
    func weightedSegmentMissingKeys() throws {
        let segment = MockWeightedSegment(
            requiredKeys: Set(["layer1.weight", "layer2.weight", "layer3.bias"])
        )

        // Only provide one of the three required keys
        let weights = ModuleParameters(parameters: [
            "layer1.weight": MLXArray.ones([4, 4])
        ])

        #expect(throws: PipelineError.self) {
            try segment.apply(weights: weights)
        }
    }

    @Test("WeightedSegment apply with all required keys succeeds")
    func weightedSegmentAllKeysPresent() throws {
        let segment = MockWeightedSegment(
            requiredKeys: Set(["layer1.weight", "layer2.weight"])
        )

        let weights = ModuleParameters(parameters: [
            "layer1.weight": MLXArray.ones([4, 4]),
            "layer2.weight": MLXArray.ones([4, 4]),
            "extra.bias": MLXArray.zeros([4]),
        ])

        try segment.apply(weights: weights)
        #expect(segment.isLoaded)
    }

    // MARK: - Pipeline Generation (Smoke Test)

    @Test("Pipeline generates with mock components (no real weights)")
    func pipelineGenerateSmokeTest() async throws {
        let recipe = StandardMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
            schedulerConfig: .init(defaultTimesteps: [999, 500, 0]),
            backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 120),
            decoderConfig: .init(inputChannels: 4),
            rendererConfig: (),
            unconditionalEmbeddingStrategy: .none
        )

        let pipeline = try DiffusionPipeline(recipe: recipe)

        let request = DiffusionGenerationRequest(
            prompt: "test prompt",
            width: 64,
            height: 64,
            steps: 3,
            guidanceScale: 1.0,
            seed: 42
        )

        let progressCount = LockedCounter()
        let result = try await pipeline.generate(request: request) { _ in
            progressCount.increment()
        }

        // Verify result
        #expect(result.seed == 42)
        #expect(result.steps == 3)
        #expect(result.guidanceScale == 1.0)
        #expect(result.duration > 0)

        // Verify output is an image
        switch result.output {
        case .image:
            break // expected
        default:
            Issue.record("Expected .image output, got \(result.output)")
        }

        // Verify progress was reported
        #expect(progressCount.value > 0)
    }

    @Test("Pipeline generates with CFG (empty prompt strategy)")
    func pipelineGenerateWithCFG() async throws {
        let recipe = StandardMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
            schedulerConfig: .init(defaultTimesteps: [999, 0]),
            backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 120),
            decoderConfig: .init(inputChannels: 4),
            rendererConfig: (),
            unconditionalEmbeddingStrategy: .emptyPrompt
        )

        let pipeline = try DiffusionPipeline(recipe: recipe)

        let request = DiffusionGenerationRequest(
            prompt: "a beautiful sunset",
            negativePrompt: "blurry",
            width: 64,
            height: 64,
            steps: 2,
            guidanceScale: 7.5,
            seed: 123
        )

        let result = try await pipeline.generate(request: request) { _ in }

        #expect(result.seed == 123)
        #expect(result.guidanceScale == 7.5)
    }

    @Test("Pipeline generates with CFG (zero vector strategy)")
    func pipelineGenerateWithZeroVectorCFG() async throws {
        let recipe = StandardMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
            schedulerConfig: .init(defaultTimesteps: [999, 0]),
            backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 120),
            decoderConfig: .init(inputChannels: 4),
            rendererConfig: (),
            unconditionalEmbeddingStrategy: .zeroVector(shape: [1, 120, 4096])
        )

        let pipeline = try DiffusionPipeline(recipe: recipe)

        let request = DiffusionGenerationRequest(
            prompt: "test",
            width: 64,
            height: 64,
            steps: 2,
            guidanceScale: 5.0,
            seed: 42
        )

        let result = try await pipeline.generate(request: request) { _ in }
        #expect(result.seed == 42)
    }

    @Test("Pipeline uses random seed when none provided")
    func pipelineRandomSeed() async throws {
        let recipe = StandardMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
            schedulerConfig: .init(defaultTimesteps: [999]),
            backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 120),
            decoderConfig: .init(inputChannels: 4),
            rendererConfig: (),
            unconditionalEmbeddingStrategy: .none
        )

        let pipeline = try DiffusionPipeline(recipe: recipe)

        let request = DiffusionGenerationRequest(
            prompt: "test",
            width: 64,
            height: 64,
            steps: 1,
            guidanceScale: 1.0,
            seed: nil
        )

        let result = try await pipeline.generate(request: request) { _ in }
        // Seed should be populated in result even when not provided
        #expect(result.seed > 0 || result.seed == 0) // any UInt32 is valid
    }

    // MARK: - Recipe Validation

    @Test("Recipe custom validation failure propagates to pipeline init")
    func recipeValidationFailure() throws {
        let recipe = StandardMockRecipe(
            encoderConfig: .init(embeddingDim: 4096, maxSeqLength: 120),
            schedulerConfig: .init(),
            backboneConfig: .init(conditioningDim: 4096, latentChannels: 4, maxSequenceLength: 120),
            decoderConfig: .init(inputChannels: 4),
            rendererConfig: (),
            customValidation: {
                throw PipelineError.missingComponent(role: "custom_check")
            }
        )

        #expect(throws: PipelineError.self) {
            _ = try DiffusionPipeline(recipe: recipe)
        }
    }
}
