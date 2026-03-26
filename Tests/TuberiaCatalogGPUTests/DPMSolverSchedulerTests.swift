@preconcurrency import MLX
import Testing
import Tuberia

@testable import TuberiaCatalog

@Suite("DPMSolverScheduler Tests")
struct DPMSolverSchedulerTests {

  // MARK: - Schedule Generation

  @Test("configure(steps:) produces correct timestep and sigma counts")
  func timestepScheduleLength() {
    let config = DPMSolverSchedulerConfiguration()
    let scheduler = DPMSolverScheduler(configuration: config)

    let plan = scheduler.configure(steps: 20)

    #expect(plan.timesteps.count == 20)
    #expect(plan.sigmas.count == 20)
  }

  @Test("Timesteps are monotonically decreasing")
  func timestepsDecreasing() {
    let config = DPMSolverSchedulerConfiguration()
    let scheduler = DPMSolverScheduler(configuration: config)

    let plan = scheduler.configure(steps: 20)

    for i in 0..<(plan.timesteps.count - 1) {
      #expect(
        plan.timesteps[i] >= plan.timesteps[i + 1],
        "Timesteps should be decreasing: \(plan.timesteps[i]) >= \(plan.timesteps[i + 1])"
      )
    }
  }

  @Test("Sigmas are non-negative and first >= last")
  func sigmasCorrespondToTimesteps() {
    let config = DPMSolverSchedulerConfiguration()
    let scheduler = DPMSolverScheduler(configuration: config)

    let plan = scheduler.configure(steps: 10)

    for sigma in plan.sigmas {
      #expect(sigma >= 0.0, "Sigmas must be non-negative")
    }

    if let first = plan.sigmas.first, let last = plan.sigmas.last {
      #expect(first >= last, "First sigma should be >= last sigma")
    }
  }

  @Test("startTimestep truncation shortens schedule correctly")
  func startTimestepTruncation() {
    let config = DPMSolverSchedulerConfiguration()
    let scheduler = DPMSolverScheduler(configuration: config)

    let fullPlan = scheduler.configure(steps: 20)
    scheduler.reset()
    let truncatedPlan = scheduler.configure(steps: 20, startTimestep: 5)

    #expect(truncatedPlan.timesteps.count == fullPlan.timesteps.count - 5)
  }

  @Test("Cosine beta schedule produces finite non-negative sigmas")
  func cosineBetaSchedule() {
    let config = DPMSolverSchedulerConfiguration(
      betaSchedule: .cosine,
      trainTimesteps: 100
    )
    let scheduler = DPMSolverScheduler(configuration: config)

    let plan = scheduler.configure(steps: 10)

    #expect(plan.timesteps.count == 10)
    for sigma in plan.sigmas {
      #expect(sigma.isFinite, "Cosine schedule sigmas should be finite")
      #expect(sigma >= 0.0, "Sigmas must be non-negative")
    }
  }

  // MARK: - Step Function

  @Test("step() preserves tensor shape")
  func stepOutputShape() {
    let config = DPMSolverSchedulerConfiguration()
    let scheduler = DPMSolverScheduler(configuration: config)

    let plan = scheduler.configure(steps: 10)

    let output = MLXArray.ones([1, 8, 8, 4]) * 0.1
    let sample = MLXArray.ones([1, 8, 8, 4]) * 0.5

    let result = scheduler.step(
      output: output,
      timestep: plan.timesteps[0],
      sample: sample
    )

    #expect(result.shape == [1, 8, 8, 4])
  }

  @Test("Denoising trajectory changes sample over multiple steps")
  func denoisingTrajectory() {
    let config = DPMSolverSchedulerConfiguration(solverOrder: 1)
    let scheduler = DPMSolverScheduler(configuration: config)

    let plan = scheduler.configure(steps: 5)

    var sample = MLXArray.ones([1, 4, 4, 4])
    let zeroTarget = MLXArray.zeros([1, 4, 4, 4])

    for timestep in plan.timesteps {
      let noisePrediction = sample - zeroTarget
      sample = scheduler.step(output: noisePrediction, timestep: timestep, sample: sample)
      eval(sample)
    }

    let initialNorm = MLXArray.ones([1, 4, 4, 4]).sum().item(Float.self)
    let finalNorm = abs(sample).sum().item(Float.self)
    #expect(finalNorm != initialNorm, "Sample should change during denoising")
  }

  @Test("Second-order solver produces finite results across steps")
  func secondOrderMultistep() {
    let config = DPMSolverSchedulerConfiguration(solverOrder: 2)
    let scheduler = DPMSolverScheduler(configuration: config)

    let plan = scheduler.configure(steps: 5)

    var sample = MLXArray.ones([1, 4, 4, 4])

    let output1 = MLXArray.ones([1, 4, 4, 4]) * 0.1
    sample = scheduler.step(output: output1, timestep: plan.timesteps[0], sample: sample)
    eval(sample)

    let output2 = MLXArray.ones([1, 4, 4, 4]) * 0.1
    sample = scheduler.step(output: output2, timestep: plan.timesteps[1], sample: sample)
    eval(sample)

    #expect(sample.mean().item(Float.self).isFinite)
  }

  @Test("Velocity prediction type produces finite step output")
  func velocityPrediction() {
    let config = DPMSolverSchedulerConfiguration(predictionType: .velocity)
    let scheduler = DPMSolverScheduler(configuration: config)

    let plan = scheduler.configure(steps: 5)
    let output = MLXArray.ones([1, 4, 4, 4]) * 0.1
    let sample = MLXArray.ones([1, 4, 4, 4]) * 0.5

    let result = scheduler.step(output: output, timestep: plan.timesteps[0], sample: sample)
    eval(result)

    #expect(result.shape == [1, 4, 4, 4])
    #expect(result.mean().item(Float.self).isFinite)
  }

  // MARK: - Noise Addition

  @Test("addNoise attenuates signal more at higher timesteps")
  func addNoiseCorrectness() {
    let config = DPMSolverSchedulerConfiguration()
    let scheduler = DPMSolverScheduler(configuration: config)

    let clean = MLXArray.ones([1, 4, 4, 4])
    let noise = MLXArray.zeros([1, 4, 4, 4])

    let resultLowNoise = scheduler.addNoise(to: clean, noise: noise, at: 0)
    eval(resultLowNoise)
    let lowNoiseMean = resultLowNoise.mean().item(Float.self)

    let resultHighNoise = scheduler.addNoise(to: clean, noise: noise, at: 999)
    eval(resultHighNoise)
    let highNoiseMean = resultHighNoise.mean().item(Float.self)

    #expect(lowNoiseMean > 0.9, "At low timestep with zero noise, result should be close to clean")
    #expect(highNoiseMean < lowNoiseMean, "At high timestep, signal should be more attenuated")
  }

  // MARK: - State Management

  @Test("reset() allows clean re-run")
  func resetClearsState() {
    let config = DPMSolverSchedulerConfiguration(solverOrder: 2)
    let scheduler = DPMSolverScheduler(configuration: config)

    let plan = scheduler.configure(steps: 5)
    var sample = MLXArray.ones([1, 4, 4, 4])
    for timestep in plan.timesteps.prefix(3) {
      let output = MLXArray.ones([1, 4, 4, 4]) * 0.1
      sample = scheduler.step(output: output, timestep: timestep, sample: sample)
      eval(sample)
    }

    scheduler.reset()

    let plan2 = scheduler.configure(steps: 5)
    let fresh = MLXArray.ones([1, 4, 4, 4])
    let output = MLXArray.ones([1, 4, 4, 4]) * 0.1
    let result = scheduler.step(output: output, timestep: plan2.timesteps[0], sample: fresh)
    eval(result)

    #expect(result.shape == [1, 4, 4, 4])
  }
}
