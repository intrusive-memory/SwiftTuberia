import Foundation
import Testing
@preconcurrency import MLX
import MLXNN

@testable import TuberiaCatalog

// MARK: - SDXLVAEModel Internal Unit Tests
//
// These tests cover the internal model classes in SDXLVAEModel.swift that were
// the site of repeatable production crashes:
//
//   Fatal error: Index out of range — AttentionBlock.callAsFunction(_:)
//   Fatal error: [Upsample] The input should have at least 1 spatial dimension
//
// Root cause: large lazy computation graphs from GroupNorm reshape chains
// produced shapeless (ndim=0) MLXArray values under memory pressure.
// The fix added strategic eval() calls at block boundaries.
//
// These tests:
//   (a) Regress the ndim crash — every output must have ndim == 4 after eval().
//   (b) Verify shape contracts for each internal module.
//   (c) Verify output finiteness and non-NaN values.
//   (d) Cover the full SDXLVAEDecoderModel forward pass (placeholder weights).
//
// All tests use @testable import — these classes are internal to TuberiaCatalog.
// Tests are serialized so Metal GPU memory is not contended.

// MARK: - ResnetBlock2D

@Suite("ResnetBlock2D — shape contracts", .serialized)
struct ResnetBlock2DTests {

    // The first resnet in VAEMidBlock: 4 → 512 channel expansion.
    @Test("4→512 channel expansion preserves spatial dims")
    func channelExpansion4to512() {
        let block = ResnetBlock2D(inChannels: 4, outChannels: 512)
        let x = MLXArray.zeros([1, 8, 8, 4])
        let out = block(x)
        eval(out)

        #expect(out.ndim == 4, "ndim must be 4 after ResnetBlock2D (got \(out.ndim))")
        #expect(out.shape == [1, 8, 8, 512], "Expected [1, 8, 8, 512], got \(out.shape)")
    }

    // Same-channel blocks: no shortcut conv.
    @Test("512→512 same-channel block preserves shape")
    func sameChannel512() {
        let block = ResnetBlock2D(inChannels: 512, outChannels: 512)
        let x = MLXArray.zeros([1, 8, 8, 512])
        let out = block(x)
        eval(out)

        #expect(out.ndim == 4, "ndim must be 4 (got \(out.ndim))")
        #expect(out.shape == [1, 8, 8, 512], "Expected [1, 8, 8, 512], got \(out.shape)")
    }

    @Test("512→256 channel reduction preserves spatial dims")
    func channelReduction512to256() {
        let block = ResnetBlock2D(inChannels: 512, outChannels: 256)
        let x = MLXArray.zeros([1, 16, 16, 512])
        let out = block(x)
        eval(out)

        #expect(out.ndim == 4, "ndim must be 4 (got \(out.ndim))")
        #expect(out.shape == [1, 16, 16, 256], "Expected [1, 16, 16, 256], got \(out.shape)")
    }

    @Test("256→128 channel reduction")
    func channelReduction256to128() {
        let block = ResnetBlock2D(inChannels: 256, outChannels: 128)
        let x = MLXArray.zeros([1, 16, 16, 256])
        let out = block(x)
        eval(out)

        #expect(out.ndim == 4, "ndim must be 4 (got \(out.ndim))")
        #expect(out.shape == [1, 16, 16, 128], "Expected [1, 16, 16, 128], got \(out.shape)")
    }

    @Test("output is finite with zero-initialised weights")
    func finiteOutput() {
        let block = ResnetBlock2D(inChannels: 4, outChannels: 512)
        let x = MLXArray.ones([1, 4, 4, 4])
        let out = block(x)
        eval(out)

        let flat = out.reshaped([-1]).asArray(Float.self)
        #expect(!flat.contains { $0.isNaN }, "ResnetBlock2D output must not contain NaN")
        #expect(!flat.contains { $0.isInfinite }, "ResnetBlock2D output must not contain Inf")
    }

    // Regression: the silu replacement (x * sigmoid(x)) must produce rank-4 output.
    // Pre-fix this block used MLXNN.silu which could return ndim=0 under pressure.
    @Test("silu replacement (x * sigmoid(x)) does not produce ndim=0 — regression")
    func siluReplacementNotZeroDim() {
        let block = ResnetBlock2D(inChannels: 4, outChannels: 4)
        let x = MLXArray.ones([2, 6, 6, 4])
        let out = block(x)
        eval(out)

        #expect(out.ndim == 4, "silu replacement must not produce ndim=0 (got ndim=\(out.ndim))")
    }
}

// MARK: - AttentionBlock

@Suite("AttentionBlock — shape contracts and flash-attention regression", .serialized)
struct AttentionBlockTests {

    // Regression: the crash was `let b = shape[0]` when `x.shape == []`.
    // Post-fix the block calls eval(x) at the top and uses
    // MLXFast.scaledDotProductAttention (no huge intermediate matrix).
    @Test("output ndim == 4 — regression for Index-out-of-range crash")
    func outputNdimIs4() {
        let block = AttentionBlock(channels: 512)
        let x = MLXArray.zeros([1, 8, 8, 512])
        let out = block(x)
        eval(out)

        #expect(out.ndim == 4, "AttentionBlock must produce rank-4 output (got ndim=\(out.ndim))")
    }

    @Test("output shape matches input shape [B, H, W, C]")
    func outputShapeMatchesInput() {
        let block = AttentionBlock(channels: 512)
        let x = MLXArray.zeros([1, 8, 8, 512])
        let out = block(x)
        eval(out)

        #expect(out.shape == [1, 8, 8, 512], "Expected [1, 8, 8, 512], got \(out.shape)")
    }

    @Test("batch > 1 preserves shape")
    func batchGreaterThan1() {
        let block = AttentionBlock(channels: 512)
        let x = MLXArray.zeros([2, 4, 4, 512])
        let out = block(x)
        eval(out)

        #expect(out.ndim == 4, "ndim must be 4 for B>1 (got \(out.ndim))")
        #expect(out.shape == [2, 4, 4, 512], "Expected [2, 4, 4, 512], got \(out.shape)")
    }

    @Test("output is finite")
    func finiteOutput() {
        let block = AttentionBlock(channels: 512)
        let x = MLXArray.ones([1, 4, 4, 512])
        let out = block(x)
        eval(out)

        let flat = out.reshaped([-1]).asArray(Float.self)
        #expect(!flat.contains { $0.isNaN }, "AttentionBlock output must not contain NaN")
        #expect(!flat.contains { $0.isInfinite }, "AttentionBlock output must not contain Inf")
    }
}

// MARK: - Upsample2D

@Suite("Upsample2D — spatial doubling", .serialized)
struct Upsample2DTests {

    @Test("2x nearest-neighbor upsampling doubles spatial dimensions")
    func spatialDoubling() {
        let up = Upsample2D(channels: 128)
        let x = MLXArray.zeros([1, 8, 8, 128])
        let out = up(x)
        eval(out)

        #expect(out.ndim == 4, "Upsample2D must produce rank-4 output (got \(out.ndim))")
        #expect(out.shape == [1, 16, 16, 128], "Expected [1, 16, 16, 128], got \(out.shape)")
    }

    @Test("upsampling preserves channel count")
    func channelPreservation() {
        let up = Upsample2D(channels: 512)
        let x = MLXArray.zeros([1, 4, 4, 512])
        let out = up(x)
        eval(out)

        #expect(out.shape[3] == 512, "Channel count must be preserved")
    }

    @Test("upsampling with batch > 1")
    func batchUpsampling() {
        let up = Upsample2D(channels: 256)
        let x = MLXArray.zeros([2, 8, 8, 256])
        let out = up(x)
        eval(out)

        #expect(out.shape == [2, 16, 16, 256], "Expected [2, 16, 16, 256], got \(out.shape)")
    }

    @Test("output is finite")
    func finiteOutput() {
        let up = Upsample2D(channels: 64)
        let x = MLXArray.ones([1, 4, 4, 64])
        let out = up(x)
        eval(out)

        let flat = out.reshaped([-1]).asArray(Float.self)
        #expect(!flat.contains { $0.isNaN }, "Upsample2D must not produce NaN")
        #expect(!flat.contains { $0.isInfinite }, "Upsample2D must not produce Inf")
    }
}

// MARK: - VAEMidBlock

@Suite("VAEMidBlock — shape contract", .serialized)
struct VAEMidBlockTests {

    // The mid-block is the first block after postQuantConv that crashed.
    @Test("4→512 expansion produces [B, H, W, 512]")
    func channelExpansionShape() {
        let block = VAEMidBlock(inChannels: 4, outChannels: 512)
        let x = MLXArray.zeros([1, 8, 8, 4])
        let out = block(x)
        eval(out)

        #expect(out.ndim == 4, "VAEMidBlock must produce rank-4 output (got \(out.ndim))")
        #expect(out.shape == [1, 8, 8, 512], "Expected [1, 8, 8, 512], got \(out.shape)")
    }

    @Test("output is finite for zero input")
    func finiteOutput() {
        let block = VAEMidBlock(inChannels: 4, outChannels: 512)
        let x = MLXArray.zeros([1, 4, 4, 4])
        let out = block(x)
        eval(out)

        let flat = out.reshaped([-1]).asArray(Float.self)
        #expect(!flat.contains { $0.isNaN }, "VAEMidBlock must not produce NaN")
        #expect(!flat.contains { $0.isInfinite }, "VAEMidBlock must not produce Inf")
    }

    // Regression: the AttentionBlock inside VAEMidBlock was crashing.
    // With eval(h) added between resnets[0] and attention, this must pass.
    @Test("AttentionBlock inside mid-block does not crash — regression for ndim=0 crash")
    func attentionInsideMidBlockNocrash() {
        let block = VAEMidBlock(inChannels: 4, outChannels: 512)
        // Use a spatial size that exercises a non-trivial lazy graph.
        let x = MLXArray.zeros([1, 16, 16, 4])
        let out = block(x)
        eval(out)

        #expect(out.shape == [1, 16, 16, 512], "VAEMidBlock with 16×16 spatial must not crash and must output [1, 16, 16, 512]")
    }
}

// MARK: - VAEUpBlock

@Suite("VAEUpBlock — shape contracts", .serialized)
struct VAEUpBlockTests {

    @Test("block with upsample doubles spatial dims")
    func withUpsampleDoublesSpace() {
        let block = VAEUpBlock(inChannels: 512, outChannels: 512, numResnetBlocks: 3, addUpsample: true)
        let x = MLXArray.zeros([1, 8, 8, 512])
        let out = block(x)
        eval(out)

        #expect(out.ndim == 4, "VAEUpBlock must produce rank-4 output (got \(out.ndim))")
        #expect(out.shape == [1, 16, 16, 512], "With upsample, spatial dims must double: expected [1, 16, 16, 512], got \(out.shape)")
    }

    @Test("block without upsample preserves spatial dims")
    func withoutUpsamplePreservesSpace() {
        let block = VAEUpBlock(inChannels: 256, outChannels: 128, numResnetBlocks: 3, addUpsample: false)
        let x = MLXArray.zeros([1, 8, 8, 256])
        let out = block(x)
        eval(out)

        #expect(out.ndim == 4, "VAEUpBlock (no upsample) must produce rank-4 output (got \(out.ndim))")
        #expect(out.shape == [1, 8, 8, 128], "Without upsample, spatial dims preserved: expected [1, 8, 8, 128], got \(out.shape)")
    }

    @Test("512→256 channel reduction with upsample")
    func channelReductionWithUpsample() {
        let block = VAEUpBlock(inChannels: 512, outChannels: 256, numResnetBlocks: 3, addUpsample: true)
        let x = MLXArray.zeros([1, 8, 8, 512])
        let out = block(x)
        eval(out)

        #expect(out.shape == [1, 16, 16, 256], "Expected [1, 16, 16, 256], got \(out.shape)")
    }

    @Test("output is finite")
    func finiteOutput() {
        let block = VAEUpBlock(inChannels: 512, outChannels: 512, numResnetBlocks: 1, addUpsample: true)
        let x = MLXArray.ones([1, 4, 4, 512])
        let out = block(x)
        eval(out)

        let flat = out.reshaped([-1]).asArray(Float.self)
        #expect(!flat.contains { $0.isNaN }, "VAEUpBlock must not produce NaN")
        #expect(!flat.contains { $0.isInfinite }, "VAEUpBlock must not produce Inf")
    }

    // Regression: Upsample2D was crashing with ndim=0 input from a ResnetBlock
    // lazy graph. The eval(h) before upsample inside VAEUpBlock must prevent this.
    @Test("eval() before Upsample prevents ndim=0 crash — regression")
    func evalBeforeUpsamplePreventsCrash() {
        // Stack two blocks to build a non-trivial lazy graph, then upsample.
        let block = VAEUpBlock(inChannels: 512, outChannels: 512, numResnetBlocks: 3, addUpsample: true)
        // 16×16 gives enough lazy graph depth to stress the eval boundary.
        let x = MLXArray.zeros([1, 16, 16, 512])
        let out = block(x)
        eval(out)

        #expect(out.shape == [1, 32, 32, 512], "Regression: Upsample must not crash with ndim=0 input; expected [1, 32, 32, 512], got \(out.shape)")
    }
}

// MARK: - SDXLVAEDecoderModel

@Suite("SDXLVAEDecoderModel — full forward pass", .serialized)
struct SDXLVAEDecoderModelTests {

    // Full pipeline shape contract (no real weights — all zero-initialised).
    // The decoder takes [B, H/8, W/8, 4] → [B, H, W, 3].
    @Test("forward pass produces [1, 64, 64, 3] from [1, 8, 8, 4] latent")
    func forwardPassShape8x8() throws {
        let model = SDXLVAEDecoderModel()
        let latents = MLXArray.zeros([1, 8, 8, 4])
        let out = model(latents)
        eval(out)

        #expect(out.ndim == 4, "Full forward pass must produce rank-4 output (got \(out.ndim))")
        #expect(out.shape == [1, 64, 64, 3], "Expected [1, 64, 64, 3], got \(out.shape)")
    }

    @Test("forward pass produces [1, 128, 128, 3] from [1, 16, 16, 4] latent")
    func forwardPassShape16x16() throws {
        let model = SDXLVAEDecoderModel()
        let latents = MLXArray.zeros([1, 16, 16, 4])
        let out = model(latents)
        eval(out)

        #expect(out.shape == [1, 128, 128, 3], "Expected [1, 128, 128, 3], got \(out.shape)")
    }

    @Test("forward pass output is finite with all-zero weights")
    func forwardPassFinite() throws {
        let model = SDXLVAEDecoderModel()
        let latents = MLXArray.zeros([1, 8, 8, 4])
        let out = model(latents)
        eval(out)

        let flat = out.reshaped([-1]).asArray(Float.self)
        #expect(!flat.contains { $0.isNaN }, "Forward pass must not produce NaN")
        #expect(!flat.contains { $0.isInfinite }, "Forward pass must not produce Inf")
    }

    // The crash scenario: 1024×1024 image → 128×128 latent.
    // Running the full up-block chain at this size is slow, but the
    // mid-block crash at 128×128 can be tested without all up-blocks.
    // We test through VAEMidBlock directly with 128×128 to avoid full cost.
    @Test("VAEMidBlock survives 128×128 latent spatial size — 1024×1024 image regression")
    func midBlockAt128x128() throws {
        let block = VAEMidBlock(inChannels: 4, outChannels: 512)
        // This is the exact spatial size for a 1024×1024 image (1024/8 = 128).
        let latents = MLXArray.zeros([1, 128, 128, 4])
        let out = block(latents)
        eval(out)

        #expect(out.ndim == 4, "VAEMidBlock at 128×128 must not crash (got ndim=\(out.ndim))")
        #expect(out.shape == [1, 128, 128, 512], "Expected [1, 128, 128, 512], got \(out.shape)")
    }
}
