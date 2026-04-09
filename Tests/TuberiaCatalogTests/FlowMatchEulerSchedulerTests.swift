import Foundation
import Testing
@preconcurrency import MLX

@testable import TuberiaCatalog

// MARK: - FlowMatchEulerScheduler CPU Unit Tests
//
// This file closes the gap that existed before Sortie 2: no CPU-only unit tests
// covered FlowMatchEulerScheduler. The GPU test suite (TuberiaCatalogGPUTests)
// requires Metal and cannot run in headless/CI environments without a GPU. These
// tests run entirely on CPU (no Metal, no real weights) and exercise:
//
//   (a) Schedule generation contracts: step count, sigma range [~1.0, ~0.0],
//       monotonic decrease, and sensitivity to the numSteps parameter.
//   (b) step() shape preservation and the zero-velocity identity contract.
//   (c) addNoise() shape preservation.
//
// Flow-matching sigma contract (shift=1.0):
//   sigmas[0] ≈ 1.0  (pure noise)
//   sigmas[last] ≈ 0.0  (clean data)
//
// eval() is called before every assertion that reads an MLXArray value so that
// MLX lazy computation is flushed before the assertion fires.
//
// Pattern: eval() → #expect(...)
//          eval() flushes lazy MLX graph; without eval() the .item() or .shape
//          access may return stale or shapeless values. Every test that reads
//          a scalar or compares element values uses eval() before assertion.

@Suite("FlowMatchEulerConfigurationTests", .serialized)
struct FlowMatchEulerConfigurationTests {

  // MARK: configureProducesCorrectStepCount

  @Test("configure(steps:) returns a plan with the requested number of timesteps")
  func configureProducesCorrectStepCount() {
    let config = FlowMatchEulerSchedulerConfiguration(shift: 1.0)
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let n = 20
    let plan = scheduler.configure(steps: n)

    // FlowMatchEuler produces n+1 sigmas (fencepost) and n timesteps
    #expect(plan.timesteps.count == n, "Expected \(n) timesteps, got \(plan.timesteps.count)")
    #expect(plan.sigmas.count == n + 1, "Expected \(n + 1) sigmas, got \(plan.sigmas.count)")
  }

  // MARK: sigmasSpanFullRange

  @Test("First sigma ≈ 1.0 and last sigma ≈ 0.0 with shift=1.0")
  func sigmasSpanFullRange() {
    let config = FlowMatchEulerSchedulerConfiguration(shift: 1.0)
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let plan = scheduler.configure(steps: 20)

    guard let firstSigma = plan.sigmas.first, let lastSigma = plan.sigmas.last else {
      Issue.record("Sigma array is unexpectedly empty")
      return
    }

    #expect(
      firstSigma > 0.9,
      "First sigma (\(firstSigma)) should be close to 1.0 (> 0.9)"
    )
    #expect(
      lastSigma < 0.1,
      "Last sigma (\(lastSigma)) should be close to 0.0 (< 0.1)"
    )
  }

  // MARK: sigmasAreMonotonicallyDecreasing

  @Test("Sigmas are monotonically non-increasing across the full schedule")
  func sigmasAreMonotonicallyDecreasing() {
    let config = FlowMatchEulerSchedulerConfiguration(shift: 1.0)
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let plan = scheduler.configure(steps: 20)

    for i in 0..<(plan.sigmas.count - 1) {
      #expect(
        plan.sigmas[i] >= plan.sigmas[i + 1],
        "sigma[\(i)] (\(plan.sigmas[i])) should be >= sigma[\(i+1)] (\(plan.sigmas[i+1]))"
      )
    }
  }

  // MARK: differentStepCountsProduceDifferentSchedules

  @Test("Schedules with different step counts produce different sigma arrays")
  func differentStepCountsProduceDifferentSchedules() {
    let config = FlowMatchEulerSchedulerConfiguration(shift: 1.0)
    let schedulerA = FlowMatchEulerScheduler(configuration: config)
    let schedulerB = FlowMatchEulerScheduler(configuration: config)

    let plan10 = schedulerA.configure(steps: 10)
    let plan20 = schedulerB.configure(steps: 20)

    // Different step counts must produce different sigma counts
    #expect(
      plan10.sigmas.count != plan20.sigmas.count,
      "10-step and 20-step schedules should have different sigma counts"
    )

    // The step sizes (sigma differences) should also differ
    let step10 = plan10.sigmas[0] - plan10.sigmas[1]
    let step20 = plan20.sigmas[0] - plan20.sigmas[1]
    #expect(
      abs(step10 - step20) > 1e-5,
      "Step sizes should differ between N=10 (\(step10)) and N=20 (\(step20)) schedules"
    )
  }
}

@Suite("FlowMatchEulerStepTests", .serialized)
struct FlowMatchEulerStepTests {

  // MARK: stepPreservesShape

  @Test("step() output has the same shape as the input sample")
  func stepPreservesShape() throws {
    let config = FlowMatchEulerSchedulerConfiguration(shift: 1.0)
    let scheduler = FlowMatchEulerScheduler(configuration: config)
    let plan = scheduler.configure(steps: 10)

    let velocity = MLXArray.ones([1, 8, 8, 4]) * 0.1
    let sample = MLXArray.ones([1, 8, 8, 4]) * 0.5

    let result = try scheduler.step(
      output: velocity,
      timestep: plan.timesteps[0],
      sample: sample
    )
    eval(result)

    #expect(
      result.shape == [1, 8, 8, 4],
      "Output shape \(result.shape) should equal input shape [1, 8, 8, 4]"
    )
  }

  // MARK: stepWithZeroVelocityIsIdentity

  @Test("step() with zero velocity returns output equal to input (identity)")
  func stepWithZeroVelocityIsIdentity() throws {
    let config = FlowMatchEulerSchedulerConfiguration(shift: 1.0)
    let scheduler = FlowMatchEulerScheduler(configuration: config)
    let plan = scheduler.configure(steps: 10)

    // sample = arbitrary non-zero values
    let sample = MLXArray.ones([1, 4, 4, 4]) * 0.42
    // velocity = all zeros → dt * 0 = 0 → output should equal sample
    let zeroVelocity = MLXArray.zeros([1, 4, 4, 4])

    let result = try scheduler.step(
      output: zeroVelocity,
      timestep: plan.timesteps[0],
      sample: sample
    )
    eval(result)
    eval(sample)

    // The Euler step is: result = sample + dt * velocity = sample + dt * 0 = sample
    let maxDiff = (result - sample).abs().max().item(Float.self)
    #expect(
      maxDiff < 1e-5,
      "With zero velocity, step() should return input unchanged. Max diff: \(maxDiff)"
    )
  }

  // MARK: addNoisePreservesShape

  @Test("addNoise() output has the same shape as the input sample")
  func addNoisePreservesShape() {
    let config = FlowMatchEulerSchedulerConfiguration(shift: 1.0)
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let sample = MLXArray.ones([2, 16, 16, 8])
    let noise = MLXArray.zeros([2, 16, 16, 8])

    let result = scheduler.addNoise(to: sample, noise: noise, at: 500)
    eval(result)

    #expect(
      result.shape == [2, 16, 16, 8],
      "addNoise() shape \(result.shape) should equal input shape [2, 16, 16, 8]"
    )
  }
}
