@preconcurrency import MLX
@testable import Tuberia

/// Mock Scheduler with configurable behavior for testing.
public final class MockScheduler: Scheduler, @unchecked Sendable {
    public struct Config: Sendable {
        public let defaultTimesteps: [Int]

        public init(defaultTimesteps: [Int] = [999, 750, 500, 250, 0]) {
            self.defaultTimesteps = defaultTimesteps
        }
    }

    public typealias Configuration = Config

    private let configuration: Configuration
    public var configureCallCount: Int = 0
    public var stepCallCount: Int = 0
    public var resetCallCount: Int = 0
    public var addNoiseCallCount: Int = 0

    public required init(configuration: Configuration) {
        self.configuration = configuration
    }

    // MARK: - Scheduler

    public func configure(steps: Int, startTimestep: Int?) -> SchedulerPlan {
        configureCallCount += 1
        var timesteps = configuration.defaultTimesteps
        if timesteps.count > steps {
            timesteps = Array(timesteps.prefix(steps))
        }
        if let start = startTimestep, start > 0, start < timesteps.count {
            timesteps = Array(timesteps.dropFirst(start))
        }
        let sigmas = timesteps.map { Float($0) / 1000.0 }
        return SchedulerPlan(timesteps: timesteps, sigmas: sigmas)
    }

    public func step(output: MLXArray, timestep: Int, sample: MLXArray) -> MLXArray {
        stepCallCount += 1
        // Simple mock: return sample with slight modification
        return sample - output * 0.1
    }

    public func addNoise(to sample: MLXArray, noise: MLXArray, at timestep: Int) -> MLXArray {
        addNoiseCallCount += 1
        let factor = Float(timestep) / 1000.0
        return sample * (1.0 - factor) + noise * factor
    }

    public func reset() {
        resetCallCount += 1
    }
}
