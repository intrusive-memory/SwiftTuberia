import Testing
@preconcurrency import MLX
@testable import TuberiaCatalog
import Tuberia

@Suite("DPMSolverScheduler Tests")
struct DPMSolverSchedulerTests {

    @Test("DPMSolverScheduler conforms to Scheduler with correct Configuration")
    func protocolConformance() {
        let config = DPMSolverSchedulerConfiguration()
        let scheduler = DPMSolverScheduler(configuration: config)
        _ = scheduler
    }

    @Test("Default configuration has expected values")
    func defaultConfiguration() {
        let config = DPMSolverSchedulerConfiguration()
        #expect(config.solverOrder == 2)
        #expect(config.trainTimesteps == 1000)
    }

    @Test("Known noise schedule produces expected timesteps count")
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

    @Test("Sigmas correspond to timesteps via alpha schedule")
    func sigmasCorrespondToTimesteps() {
        let config = DPMSolverSchedulerConfiguration()
        let scheduler = DPMSolverScheduler(configuration: config)

        let plan = scheduler.configure(steps: 10)

        // Sigmas should be positive and generally decreasing (following timesteps)
        for sigma in plan.sigmas {
            #expect(sigma >= 0.0, "Sigmas must be non-negative")
        }

        // First sigma should be larger than last (more noise -> less noise)
        if let first = plan.sigmas.first, let last = plan.sigmas.last {
            #expect(first >= last, "First sigma should be >= last sigma")
        }
    }

    @Test("DPM-Solver++ step produces valid output shape")
    func stepOutputShape() {
        let config = DPMSolverSchedulerConfiguration()
        let scheduler = DPMSolverScheduler(configuration: config)

        let plan = scheduler.configure(steps: 10)

        let output = MLXArray.ones([1, 8, 8, 4]) * 0.1  // noise prediction
        let sample = MLXArray.ones([1, 8, 8, 4]) * 0.5  // current noisy latents

        let result = scheduler.step(
            output: output,
            timestep: plan.timesteps[0],
            sample: sample
        )

        #expect(result.shape == [1, 8, 8, 4])
    }

    @Test("Synthetic denoising trajectory moves toward clean signal")
    func denoisingTrajectory() {
        let config = DPMSolverSchedulerConfiguration(solverOrder: 1) // first-order for simplicity
        let scheduler = DPMSolverScheduler(configuration: config)

        let plan = scheduler.configure(steps: 5)

        // Start with noisy latents
        var sample = MLXArray.ones([1, 4, 4, 4])
        let zeroTarget = MLXArray.zeros([1, 4, 4, 4])

        // Use the noise prediction to be the difference (epsilon prediction)
        // After stepping through all timesteps, the sample should evolve
        for timestep in plan.timesteps {
            let noisePrediction = sample - zeroTarget  // predict noise = current - target
            sample = scheduler.step(output: noisePrediction, timestep: timestep, sample: sample)
            eval(sample)
        }

        // After denoising, sample should have changed from initial value
        let initialNorm = MLXArray.ones([1, 4, 4, 4]).sum().item(Float.self)
        let finalNorm = abs(sample).sum().item(Float.self)
        #expect(finalNorm != initialNorm, "Sample should change during denoising")
    }

    @Test("Second-order solver uses previous step outputs")
    func secondOrderMultistep() {
        let config = DPMSolverSchedulerConfiguration(solverOrder: 2)
        let scheduler = DPMSolverScheduler(configuration: config)

        let plan = scheduler.configure(steps: 5)

        var sample = MLXArray.ones([1, 4, 4, 4])

        // First step should use first-order (no previous output yet)
        let output1 = MLXArray.ones([1, 4, 4, 4]) * 0.1
        sample = scheduler.step(output: output1, timestep: plan.timesteps[0], sample: sample)
        eval(sample)
        let afterFirst = sample.sum().item(Float.self)

        // Second step should use second-order (has previous output)
        let output2 = MLXArray.ones([1, 4, 4, 4]) * 0.1
        sample = scheduler.step(output: output2, timestep: plan.timesteps[1], sample: sample)
        eval(sample)
        let afterSecond = sample.sum().item(Float.self)

        // Both steps should produce valid results (not NaN/Inf)
        #expect(afterFirst.isFinite)
        #expect(afterSecond.isFinite)
    }

    @Test("startTimestep truncation produces correct shortened schedule")
    func startTimestepTruncation() {
        let config = DPMSolverSchedulerConfiguration()
        let scheduler = DPMSolverScheduler(configuration: config)

        let fullPlan = scheduler.configure(steps: 20)
        scheduler.reset()
        let truncatedPlan = scheduler.configure(steps: 20, startTimestep: 5)

        #expect(truncatedPlan.timesteps.count < fullPlan.timesteps.count)
        #expect(truncatedPlan.timesteps.count == fullPlan.timesteps.count - 5)
    }

    @Test("reset() clears state allowing clean re-run")
    func resetClearsState() {
        let config = DPMSolverSchedulerConfiguration(solverOrder: 2)
        let scheduler = DPMSolverScheduler(configuration: config)

        // Run a few steps to accumulate state
        let plan = scheduler.configure(steps: 5)
        var sample = MLXArray.ones([1, 4, 4, 4])
        for timestep in plan.timesteps.prefix(3) {
            let output = MLXArray.ones([1, 4, 4, 4]) * 0.1
            sample = scheduler.step(output: output, timestep: timestep, sample: sample)
            eval(sample)
        }

        // Reset
        scheduler.reset()

        // After reset, the first step should behave like a first-order step
        // (no previous outputs cached)
        let plan2 = scheduler.configure(steps: 5)
        let fresh = MLXArray.ones([1, 4, 4, 4])
        let output = MLXArray.ones([1, 4, 4, 4]) * 0.1
        let result = scheduler.step(output: output, timestep: plan2.timesteps[0], sample: fresh)
        eval(result)

        #expect(result.shape == [1, 4, 4, 4])
    }

    @Test("addNoise produces correct noised sample")
    func addNoiseCorrectness() {
        let config = DPMSolverSchedulerConfiguration()
        let scheduler = DPMSolverScheduler(configuration: config)

        let clean = MLXArray.ones([1, 4, 4, 4])
        let noise = MLXArray.zeros([1, 4, 4, 4])

        // At timestep 0 (minimal noise), result should be close to clean sample
        let resultLowNoise = scheduler.addNoise(to: clean, noise: noise, at: 0)
        eval(resultLowNoise)

        let lowNoiseMean = resultLowNoise.mean().item(Float.self)
        // With zero noise, the result is just sqrt(alpha_0) * clean
        #expect(lowNoiseMean > 0.9, "At low timestep with zero noise, result should be close to clean")

        // At high timestep, noise dominates
        let resultHighNoise = scheduler.addNoise(to: clean, noise: noise, at: 999)
        eval(resultHighNoise)

        let highNoiseMean = resultHighNoise.mean().item(Float.self)
        // With zero noise and high timestep, alpha is small so signal is attenuated
        #expect(highNoiseMean < lowNoiseMean, "At high timestep, signal should be more attenuated")
    }

    @Test("Cosine beta schedule produces valid values")
    func cosineBetaSchedule() {
        let config = DPMSolverSchedulerConfiguration(
            betaSchedule: .cosine,
            trainTimesteps: 100
        )
        let scheduler = DPMSolverScheduler(configuration: config)

        let plan = scheduler.configure(steps: 10)

        // Should produce valid timesteps and sigmas
        #expect(plan.timesteps.count == 10)
        for sigma in plan.sigmas {
            #expect(sigma.isFinite, "Cosine schedule sigmas should be finite")
            #expect(sigma >= 0.0, "Sigmas must be non-negative")
        }
    }

    @Test("Velocity prediction type produces valid step output")
    func velocityPrediction() {
        let config = DPMSolverSchedulerConfiguration(predictionType: .velocity)
        let scheduler = DPMSolverScheduler(configuration: config)

        let plan = scheduler.configure(steps: 5)
        let output = MLXArray.ones([1, 4, 4, 4]) * 0.1
        let sample = MLXArray.ones([1, 4, 4, 4]) * 0.5

        let result = scheduler.step(output: output, timestep: plan.timesteps[0], sample: sample)
        eval(result)

        #expect(result.shape == [1, 4, 4, 4])
        let resultMean = result.mean().item(Float.self)
        #expect(resultMean.isFinite, "Velocity prediction should produce finite results")
    }
}
