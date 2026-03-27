@preconcurrency import MLX
import Testing
import Tuberia

@testable import TuberiaCatalog

@Suite("FlowMatchEulerScheduler Tests")
struct FlowMatchEulerSchedulerTests {

  // MARK: - Schedule Generation

  @Test("configure(steps:) produces steps+1 sigmas and steps timesteps")
  func sigmaScheduleLength() {
    let config = FlowMatchEulerSchedulerConfiguration()
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let plan = scheduler.configure(steps: 20)

    #expect(plan.sigmas.count == 21)
    #expect(plan.timesteps.count == 20)
  }

  @Test("Sigma schedule spans ~1.0 to ~0.0 with shift=1.0")
  func sigmaScheduleRange() {
    let config = FlowMatchEulerSchedulerConfiguration(shift: 1.0)
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let plan = scheduler.configure(steps: 20)

    #expect(plan.sigmas.first! > 0.9, "First sigma should be close to 1.0")
    #expect(plan.sigmas.last! < 0.1, "Last sigma should be close to 0.0")
  }

  @Test("Sigmas are monotonically decreasing")
  func sigmasDecreasing() {
    let config = FlowMatchEulerSchedulerConfiguration()
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let plan = scheduler.configure(steps: 20)

    for i in 0..<(plan.sigmas.count - 1) {
      #expect(
        plan.sigmas[i] >= plan.sigmas[i + 1],
        "Sigmas should be decreasing: \(plan.sigmas[i]) >= \(plan.sigmas[i + 1])"
      )
    }
  }

  @Test("Shift > 1 changes sigma distribution in the middle range")
  func shiftAdjustsSigmas() {
    let noShiftScheduler = FlowMatchEulerScheduler(
      configuration: FlowMatchEulerSchedulerConfiguration(shift: 1.0))
    let shiftedScheduler = FlowMatchEulerScheduler(
      configuration: FlowMatchEulerSchedulerConfiguration(shift: 3.0))

    let noShiftPlan = noShiftScheduler.configure(steps: 10)
    let shiftedPlan = shiftedScheduler.configure(steps: 10)

    #expect(noShiftPlan.sigmas.count == shiftedPlan.sigmas.count)

    var hasDifference = false
    for i in 1..<(noShiftPlan.sigmas.count - 1) {
      if abs(noShiftPlan.sigmas[i] - shiftedPlan.sigmas[i]) > 0.01 {
        hasDifference = true
        break
      }
    }
    #expect(hasDifference, "Shifted sigmas should differ from unshifted in the middle range")
  }

  // MARK: - Step Function

  @Test("Euler step preserves tensor shape")
  func eulerStepShape() throws {
    let config = FlowMatchEulerSchedulerConfiguration()
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let plan = scheduler.configure(steps: 10)

    let velocity = MLXArray.ones([1, 8, 8, 4]) * 0.1
    let sample = MLXArray.ones([1, 8, 8, 4]) * 0.5

    let result = try scheduler.step(
      output: velocity,
      timestep: plan.timesteps[0],
      sample: sample
    )

    #expect(result.shape == [1, 8, 8, 4])
  }

  @Test("Euler step changes zero-initialized sample")
  func eulerStepDirection() throws {
    let config = FlowMatchEulerSchedulerConfiguration()
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let plan = scheduler.configure(steps: 10)

    let velocity = MLXArray.ones([1, 4, 4, 4])
    let sample = MLXArray.zeros([1, 4, 4, 4])

    let result = try scheduler.step(
      output: velocity,
      timestep: plan.timesteps[0],
      sample: sample
    )
    eval(result)

    let resultMean = result.mean().item(Float.self)
    #expect(resultMean.isFinite)
    #expect(resultMean != 0.0, "Step should change the sample")
  }

  @Test("Full trajectory produces finite results")
  func fullTrajectory() throws {
    let config = FlowMatchEulerSchedulerConfiguration()
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let plan = scheduler.configure(steps: 10)

    var sample = MLXArray.ones([1, 4, 4, 4])

    for timestep in plan.timesteps {
      let velocity = MLXArray.ones([1, 4, 4, 4]) * 0.1
      sample = try scheduler.step(output: velocity, timestep: timestep, sample: sample)
      eval(sample)
    }

    #expect(sample.mean().item(Float.self).isFinite)
  }

  // MARK: - Noise Addition

  @Test("addNoise follows flow matching interpolation formula")
  func addNoiseFlowMatching() {
    let config = FlowMatchEulerSchedulerConfiguration()
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let clean = MLXArray.ones([1, 4, 4, 4])
    let noise = MLXArray.zeros([1, 4, 4, 4])

    // x_t = (1 - sigma) * x_0 + sigma * noise
    // With noise = 0: at t=100 sigma=0.1, result = 0.9; at t=900 sigma=0.9, result = 0.1
    let resultLow = scheduler.addNoise(to: clean, noise: noise, at: 100)
    eval(resultLow)
    let resultHigh = scheduler.addNoise(to: clean, noise: noise, at: 900)
    eval(resultHigh)

    #expect(abs(resultLow.mean().item(Float.self) - 0.9) < 0.01)
    #expect(abs(resultHigh.mean().item(Float.self) - 0.1) < 0.01)
  }

  // MARK: - State Management

  @Test("reset() allows clean reconfiguration")
  func resetClearsState() throws {
    let config = FlowMatchEulerSchedulerConfiguration()
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let plan = scheduler.configure(steps: 5)
    let velocity = MLXArray.ones([1, 4, 4, 4]) * 0.1
    let sample = MLXArray.ones([1, 4, 4, 4])
    let _ = try scheduler.step(output: velocity, timestep: plan.timesteps[0], sample: sample)

    scheduler.reset()

    let plan2 = scheduler.configure(steps: 10)
    #expect(plan2.timesteps.count == 10)
  }

  // MARK: - Error Handling

  @Test("step() without prior configure() throws descriptive error")
  func stepWithoutConfigureThrows() {
    let config = FlowMatchEulerSchedulerConfiguration()
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let velocity = MLXArray.ones([1, 4, 4, 4])
    let sample = MLXArray.ones([1, 4, 4, 4])

    #expect(throws: PipelineError.self) {
      try scheduler.step(
        output: velocity,
        timestep: 100,
        sample: sample
      )
    }
  }

  @Test("step() with out-of-plan timestep snaps to nearest valid timestep")
  func stepWithOutOfPlanTimestepSnapsToNearest() throws {
    let config = FlowMatchEulerSchedulerConfiguration()
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let plan = scheduler.configure(steps: 10)

    let velocity = MLXArray.ones([1, 4, 4, 4]) * 0.1
    let sample = MLXArray.ones([1, 4, 4, 4])

    // Use a timestep that is NOT in the plan
    let validTimestep = plan.timesteps[0]
    let invalidTimestep = validTimestep + 50  // Likely not in the plan

    // This should NOT throw; it should snap to the nearest timestep
    let result = try scheduler.step(
      output: velocity,
      timestep: invalidTimestep,
      sample: sample
    )

    #expect(result.shape == sample.shape, "Result should have the same shape as input sample")

    eval(result)
    let resultMean = result.mean().item(Float.self)
    #expect(resultMean.isFinite, "Result should contain finite values")
    #expect(resultMean != sample.mean().item(Float.self), "Result should be different from input sample (velocity applied)")
  }

  @Test("step() with out-of-plan timestep produces consistent results")
  func stepWithOutOfPlanTimestepIsConsistent() throws {
    let config = FlowMatchEulerSchedulerConfiguration()
    let scheduler = FlowMatchEulerScheduler(configuration: config)

    let plan = scheduler.configure(steps: 10)

    let velocity = MLXArray.ones([1, 4, 4, 4]) * 0.1
    let sample = MLXArray.ones([1, 4, 4, 4])

    // Use two different out-of-plan timesteps that snap to the same nearest timestep
    let invalidTimestep1 = plan.timesteps[0] + 5
    let invalidTimestep2 = plan.timesteps[0] + 8

    let result1 = try scheduler.step(
      output: velocity,
      timestep: invalidTimestep1,
      sample: sample
    )

    let result2 = try scheduler.step(
      output: velocity,
      timestep: invalidTimestep2,
      sample: sample
    )

    eval(result1)
    eval(result2)

    // Both should snap to the same nearest timestep, so results should be close
    let diff = (result1 - result2).abs().max().item(Float.self)
    #expect(diff < 1e-5, "Results for nearby out-of-plan timesteps should be nearly identical")
  }
}
