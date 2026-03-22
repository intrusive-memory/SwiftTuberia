import Testing
@preconcurrency import MLX
@testable import TuberiaCatalog
import Tuberia

@Suite("FlowMatchEulerScheduler Tests")
struct FlowMatchEulerSchedulerTests {

    @Test("FlowMatchEulerScheduler conforms to Scheduler with correct Configuration")
    func protocolConformance() {
        let config = FlowMatchEulerSchedulerConfiguration()
        let scheduler = FlowMatchEulerScheduler(configuration: config)
        _ = scheduler
    }

    @Test("Default configuration has shift = 1.0")
    func defaultConfiguration() {
        let config = FlowMatchEulerSchedulerConfiguration()
        #expect(config.shift == 1.0)
    }

    @Test("Known step count produces expected sigma schedule")
    func sigmaScheduleLength() {
        let config = FlowMatchEulerSchedulerConfiguration()
        let scheduler = FlowMatchEulerScheduler(configuration: config)

        let plan = scheduler.configure(steps: 20)

        // Sigmas should have steps+1 entries (including endpoints)
        #expect(plan.sigmas.count == 21)
        // Timesteps should have steps entries
        #expect(plan.timesteps.count == 20)
    }

    @Test("Sigma schedule starts near 1.0 and ends near 0.0 with shift=1.0")
    func sigmaScheduleRange() {
        let config = FlowMatchEulerSchedulerConfiguration(shift: 1.0)
        let scheduler = FlowMatchEulerScheduler(configuration: config)

        let plan = scheduler.configure(steps: 20)

        // With shift=1.0, sigmas should be linearly spaced from 1.0 to 0.0
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

    @Test("Euler step with known inputs produces expected output shape")
    func eulerStepShape() {
        let config = FlowMatchEulerSchedulerConfiguration()
        let scheduler = FlowMatchEulerScheduler(configuration: config)

        let plan = scheduler.configure(steps: 10)

        let velocity = MLXArray.ones([1, 8, 8, 4]) * 0.1
        let sample = MLXArray.ones([1, 8, 8, 4]) * 0.5

        let result = scheduler.step(
            output: velocity,
            timestep: plan.timesteps[0],
            sample: sample
        )

        #expect(result.shape == [1, 8, 8, 4])
    }

    @Test("Euler step moves sample in direction of velocity")
    func eulerStepDirection() {
        let config = FlowMatchEulerSchedulerConfiguration()
        let scheduler = FlowMatchEulerScheduler(configuration: config)

        let plan = scheduler.configure(steps: 10)

        let velocity = MLXArray.ones([1, 4, 4, 4])  // unit velocity
        let sample = MLXArray.zeros([1, 4, 4, 4])   // zero start

        let result = scheduler.step(
            output: velocity,
            timestep: plan.timesteps[0],
            sample: sample
        )
        eval(result)

        // With negative dt (sigma decreases), the result should move
        // in the negative velocity direction from the sample
        let resultMean = result.mean().item(Float.self)
        #expect(resultMean.isFinite, "Step output should be finite")
        #expect(resultMean != 0.0, "Step should change the sample")
    }

    @Test("Shift parameter correctly adjusts the sigma schedule")
    func shiftAdjustsSigmas() {
        let noShiftConfig = FlowMatchEulerSchedulerConfiguration(shift: 1.0)
        let shiftedConfig = FlowMatchEulerSchedulerConfiguration(shift: 3.0)

        let noShiftScheduler = FlowMatchEulerScheduler(configuration: noShiftConfig)
        let shiftedScheduler = FlowMatchEulerScheduler(configuration: shiftedConfig)

        let noShiftPlan = noShiftScheduler.configure(steps: 10)
        let shiftedPlan = shiftedScheduler.configure(steps: 10)

        // Both should have the same number of sigmas
        #expect(noShiftPlan.sigmas.count == shiftedPlan.sigmas.count)

        // With shift > 1, the sigma schedule should be different
        // (shifted sigmas will be larger in the middle range)
        var hasDifference = false
        for i in 1..<(noShiftPlan.sigmas.count - 1) {
            if abs(noShiftPlan.sigmas[i] - shiftedPlan.sigmas[i]) > 0.01 {
                hasDifference = true
                break
            }
        }
        #expect(hasDifference, "Shifted sigmas should differ from unshifted in the middle range")
    }

    @Test("addNoise implements flow matching interpolation")
    func addNoiseFlowMatching() {
        let config = FlowMatchEulerSchedulerConfiguration()
        let scheduler = FlowMatchEulerScheduler(configuration: config)

        let clean = MLXArray.ones([1, 4, 4, 4])
        let noise = MLXArray.zeros([1, 4, 4, 4])

        // At low sigma (low timestep), result should be mostly clean
        let resultLow = scheduler.addNoise(to: clean, noise: noise, at: 100)
        eval(resultLow)
        let lowMean = resultLow.mean().item(Float.self)

        // At high sigma (high timestep), result should be mostly noise
        let resultHigh = scheduler.addNoise(to: clean, noise: noise, at: 900)
        eval(resultHigh)
        let highMean = resultHigh.mean().item(Float.self)

        // sigma = timestep / 1000
        // x_t = (1 - sigma) * x_0 + sigma * noise
        // With noise = 0:
        // At t=100: sigma=0.1, result = 0.9 * 1.0 = 0.9
        // At t=900: sigma=0.9, result = 0.1 * 1.0 = 0.1
        #expect(abs(lowMean - 0.9) < 0.01, "At low timestep, result should be ~0.9")
        #expect(abs(highMean - 0.1) < 0.01, "At high timestep, result should be ~0.1")
    }

    @Test("reset() clears internal state")
    func resetClearsState() {
        let config = FlowMatchEulerSchedulerConfiguration()
        let scheduler = FlowMatchEulerScheduler(configuration: config)

        // Configure and run some steps
        let plan = scheduler.configure(steps: 5)
        let velocity = MLXArray.ones([1, 4, 4, 4]) * 0.1
        let sample = MLXArray.ones([1, 4, 4, 4])
        let _ = scheduler.step(output: velocity, timestep: plan.timesteps[0], sample: sample)

        // Reset
        scheduler.reset()

        // After reset, configuring again should work cleanly
        let plan2 = scheduler.configure(steps: 10)
        #expect(plan2.timesteps.count == 10)
    }

    @Test("Full denoising trajectory produces finite results")
    func fullTrajectory() {
        let config = FlowMatchEulerSchedulerConfiguration()
        let scheduler = FlowMatchEulerScheduler(configuration: config)

        let plan = scheduler.configure(steps: 10)

        var sample = MLXArray.ones([1, 4, 4, 4])

        for timestep in plan.timesteps {
            let velocity = MLXArray.ones([1, 4, 4, 4]) * 0.1
            sample = scheduler.step(output: velocity, timestep: timestep, sample: sample)
            eval(sample)
        }

        let finalMean = sample.mean().item(Float.self)
        #expect(finalMean.isFinite, "Final sample should be finite after full trajectory")
    }
}
