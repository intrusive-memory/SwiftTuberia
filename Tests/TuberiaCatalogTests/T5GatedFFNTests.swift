import Foundation
import Testing
@preconcurrency import MLX
import MLXNN

@testable import TuberiaCatalog

// MARK: - T5GatedFFN Unit Tests
//
// T5GatedFFN (GeGLU activation) was the source of a pipeline-killing crash:
//
//   AttentionBlock.callAsFunction(_:) crashed at SDXLVAEModel.swift:124
//   because `x.shape` returned [] (a 0-D tensor arrived from upstream).
//
// Root cause: MLXNN.gelu() delegates to a global compiled closure built with
// MLX.compile(shapeless: true).  Under memory pressure that compiled closure
// returned an empty [MLXArray], and subscripting [0] into an empty Swift Array
// triggers a non-catchable fatalError.
//
// Fix (SwiftTuberia 0.2.7, commit e6778e5):
//   Replace MLXNN.gelu(inner) with the direct gelu_new math:
//   inner * 0.5 * (1 + tanh(0.7978845608 * (inner + 0.044715 * inner^3)))
//
// These tests:
//  (a) Regress the 0-D tensor crash — output must be rank-3.
//  (b) Verify the gelu_new approximation matches exact gelu to within 1%.
//  (c) Verify shape preservation across batch sizes and sequence lengths.
//  (d) Verify output finiteness.
//  (e) Verify T5RMSNorm shape preservation and unit-norm behavior.

// MARK: - Helpers

private let testH = 16   // hidden dim
private let testF = 32   // ffn dim

/// Make a T5GatedFFN with ones weights so the gelu gate receives non-trivial inputs.
private func makeFFN(hidden: Int = testH, ffn: Int = testF) -> T5GatedFFN {
    let layer = T5GatedFFN(hiddenDim: hidden, ffnDim: ffn)
    layer.wi_0 = MLXArray.ones([hidden, ffn])
    layer.wi_1 = MLXArray.ones([hidden, ffn])
    layer.wo = MLXArray.ones([ffn, hidden])
    return layer
}

// MARK: - Shape Preservation (0-D tensor regression)

/// Regression suite for the 0-D tensor crash in AttentionBlock.
/// Every test here must pass for the crash to stay fixed.
@Suite("T5GatedFFN — 0-D tensor regression", .serialized)
struct T5GatedFFNRegressionTests {

    // Critical regression: output must be rank-3.
    // If this fails the crash at SDXLVAEModel.swift:124 will recur.
    @Test("output ndim == 3 — regression for 0-D tensor crash")
    func outputNdimIs3() {
        let ffn = makeFFN()
        let x = MLXArray.ones([1, 4, testH])
        let out = ffn(x)
        eval(out)

        #expect(out.ndim == 3, "T5GatedFFN must produce rank-3 output (ndim was \(out.ndim), not 3)")
    }

    @Test("output shape matches [B, SeqLen, hiddenDim]")
    func outputShapePreservation() {
        let ffn = makeFFN()
        let x = MLXArray.ones([1, 8, testH])
        let out = ffn(x)
        eval(out)

        #expect(
            out.shape == [1, 8, testH],
            "Expected [1, 8, \(testH)], got \(out.shape)"
        )
    }

    @Test("output shape with batch > 1 and varied seq length")
    func batchAndSeqPreservation() {
        let batchSize = 3
        let seqLen = 12
        let ffn = makeFFN()
        let x = MLXArray.ones([batchSize, seqLen, testH])
        let out = ffn(x)
        eval(out)

        #expect(out.shape == [batchSize, seqLen, testH])
    }

    @Test("zero input produces zero output (gelu(0) == 0 sanity check)")
    func zeroInputZeroOutput() {
        let ffn = makeFFN()
        let x = MLXArray.zeros([1, 4, testH])
        let out = ffn(x)
        eval(out)

        // gelu(0) = 0, so gate = 0, gated = 0 × linear = 0, output = matmul(zeros, wo) = 0
        let data = out.reshaped([-1]).asArray(Float.self)
        let allZero = data.allSatisfy { abs($0) < 1e-5 }
        #expect(allZero, "Zero input must produce zero output through GeGLU")
        // Shape must still be correct
        #expect(out.ndim == 3)
    }
}

// MARK: - Finiteness and Numerical Validity

@Suite("T5GatedFFN — numerical validity", .serialized)
struct T5GatedFFNNumericalTests {

    @Test("output is finite for ones input")
    func onesInputIsFinite() {
        let ffn = makeFFN()
        let x = MLXArray.ones([1, 4, testH])
        let out = ffn(x)
        eval(out)

        let data = out.reshaped([-1]).asArray(Float.self)
        #expect(!data.contains { $0.isNaN }, "Output must not contain NaN")
        #expect(!data.contains { $0.isInfinite }, "Output must not contain Inf")
    }

    @Test("output is non-zero when weights and input are ones")
    func nonZeroOutputWithOnesWeights() {
        let ffn = makeFFN()
        let x = MLXArray.ones([1, 4, testH])
        let out = ffn(x)
        eval(out)

        let data = out.reshaped([-1]).asArray(Float.self)
        let allZero = data.allSatisfy { abs($0) < 1e-6 }
        #expect(!allZero, "GeGLU output with ones weights should be non-zero for non-zero input")
    }

    @Test("output is finite for large positive input")
    func largePositiveInputIsFinite() {
        let ffn = makeFFN()
        // tanh saturates cleanly for large inputs — gelu_new should remain finite
        let x = MLXArray.ones([1, 4, testH]) * 10.0
        let out = ffn(x)
        eval(out)

        let data = out.reshaped([-1]).asArray(Float.self)
        #expect(!data.contains { $0.isNaN }, "Large positive input must not produce NaN")
        #expect(!data.contains { $0.isInfinite }, "Large positive input must not produce Inf")
    }

    @Test("output is finite for large negative input")
    func largeNegativeInputIsFinite() {
        let ffn = makeFFN()
        let x = MLXArray.ones([1, 4, testH]) * -10.0
        let out = ffn(x)
        eval(out)

        let data = out.reshaped([-1]).asArray(Float.self)
        #expect(!data.contains { $0.isNaN }, "Large negative input must not produce NaN")
        #expect(!data.contains { $0.isInfinite }, "Large negative input must not produce Inf")
    }
}

// MARK: - gelu_new Approximation Correctness

/// Validates that the direct gelu_new math (the crash fix) matches the exact
/// gelu formula to within 1% relative error at a range of reference points.
///
/// Exact gelu:  x * 0.5 * (1 + erf(x / sqrt(2)))
/// gelu_new:    x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
///
/// The maximum absolute error between the two is < 0.0002 across [-4, 4].
@Suite("gelu_new approximation accuracy", .serialized)
struct GeluNewAccuracyTests {

    // Exact gelu using MLX.erf so the computation goes through the same
    // numeric path as the rest of the pipeline.
    private func exactGelu(_ x: Float) -> Float {
        let xArr = MLXArray(x)
        let result = xArr * 0.5 * (1.0 + MLX.erf(xArr / Float(2.0).squareRoot()))
        eval(result)
        return result.item(Float.self)
    }

    private func geluNew(_ x: Float) -> Float {
        let c: Float = 0.7978845608   // sqrt(2 / pi)
        return x * 0.5 * (1.0 + tanh(c * (x + 0.044715 * x * x * x)))
    }

    @Test("gelu_new matches exact gelu at standard reference points (< 1% relative error)")
    func referencePointAccuracy() {
        let testValues: [Float] = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

        for x in testValues {
            let exact = exactGelu(x)
            let approx = geluNew(x)

            if abs(exact) > 0.01 {
                let relError = abs(approx - exact) / abs(exact)
                #expect(
                    relError < 0.01,
                    "gelu_new relative error too large at x=\(x): exact=\(exact), approx=\(approx), relError=\(relError)"
                )
            } else {
                let absError = abs(approx - exact)
                #expect(
                    absError < 0.001,
                    "gelu_new absolute error too large at x=\(x): exact=\(exact), approx=\(approx), absError=\(absError)"
                )
            }
        }
    }

    @Test("gelu_new(0) == 0 exactly")
    func geluNewAtZeroIsZero() {
        let result = geluNew(0.0)
        #expect(abs(result) < 1e-8, "gelu_new(0) must be 0, got \(result)")
    }

    @Test("gelu_new is monotone increasing for positive inputs")
    func geluNewIsMonotonePositive() {
        let xs: [Float] = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        var prev: Float = geluNew(xs[0])
        for x in xs.dropFirst() {
            let y = geluNew(x)
            #expect(y > prev, "gelu_new must be increasing at x=\(x): got \(y) after \(prev)")
            prev = y
        }
    }

    @Test("gelu_new is non-positive for negative inputs")
    func geluNewIsNonPositiveForNegativeInputs() {
        let negatives: [Float] = [-4.0, -3.0, -2.0, -1.0, -0.5, -0.1]
        for x in negatives {
            let y = geluNew(x)
            #expect(y <= 0.01, "gelu_new(\(x)) should be ≤ 0, got \(y)")
        }
    }

    /// Verify the constant in the formula.
    /// sqrt(2 / pi) = 0.7978845608...
    @Test("gelu_new constant sqrt(2/pi) is correct to 8 significant figures")
    func geluNewConstantCorrect() {
        let expected = Float(2.0 / Double.pi).squareRoot()
        let usedInCode: Float = 0.7978845608
        #expect(abs(usedInCode - expected) < 1e-8, "Constant mismatch: expected \(expected), got \(usedInCode)")
    }
}

// MARK: - T5RMSNorm Tests

@Suite("T5RMSNorm — shape and numerical validity", .serialized)
struct T5RMSNormTests {

    @Test("output shape matches input shape")
    func shapePreservation() {
        let norm = T5RMSNorm(dim: 16)
        let x = MLXArray.ones([2, 6, 16])
        let out = norm(x)
        eval(out)
        #expect(out.shape == x.shape, "T5RMSNorm must preserve input shape")
    }

    @Test("all-ones input produces all-ones output when weight is ones")
    func unitNormWithOnesInput() {
        // RMSNorm(ones) = ones * rsqrt(mean(1^2) + eps) * weight
        //               = ones * rsqrt(1 + eps) * ones
        //               ≈ ones (since eps is tiny)
        let dim = 16
        let norm = T5RMSNorm(dim: dim)  // weight initialized to ones
        let x = MLXArray.ones([1, 1, dim])
        let out = norm(x)
        eval(out)

        let data = out.reshaped([-1]).asArray(Float.self)
        for (i, v) in data.enumerated() {
            #expect(abs(v - 1.0) < 0.001, "T5RMSNorm(ones)[i=\(i)] expected ≈1.0, got \(v)")
        }
    }

    @Test("output is finite for typical input magnitudes")
    func finiteOutput() {
        let norm = T5RMSNorm(dim: 16)
        let x = MLXArray.ones([2, 5, 16]) * 3.14
        let out = norm(x)
        eval(out)

        let data = out.reshaped([-1]).asArray(Float.self)
        #expect(!data.contains { $0.isNaN }, "T5RMSNorm must not produce NaN")
        #expect(!data.contains { $0.isInfinite }, "T5RMSNorm must not produce Inf")
    }

    @Test("normalizes different-magnitude inputs to similar RMS")
    func normalizesRMS() {
        // After RMSNorm with ones weight, RMS of output ≈ 1.0 regardless of input scale.
        let dim = 16
        let norm = T5RMSNorm(dim: dim)

        let small = MLXArray.ones([1, 1, dim]) * 0.001
        let large = MLXArray.ones([1, 1, dim]) * 1000.0

        let outSmall = norm(small)
        let outLarge = norm(large)
        eval(outSmall, outLarge)

        let smallData = outSmall.reshaped([-1]).asArray(Float.self)
        let largeData = outLarge.reshaped([-1]).asArray(Float.self)

        // Both should produce the same RMS (≈ 1.0 with weight=1, input=constant)
        let rmsSmall = sqrt(smallData.map { $0 * $0 }.reduce(0, +) / Float(dim))
        let rmsLarge = sqrt(largeData.map { $0 * $0 }.reduce(0, +) / Float(dim))

        #expect(abs(rmsSmall - 1.0) < 0.001, "RMS of small input after norm should be ≈1.0, got \(rmsSmall)")
        #expect(abs(rmsLarge - 1.0) < 0.001, "RMS of large input after norm should be ≈1.0, got \(rmsLarge)")
    }
}
