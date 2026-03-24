import Tuberia

// MARK: - FlowMatchEulerScheduler Configuration

public struct FlowMatchEulerSchedulerConfiguration: Sendable {
    /// Shift parameter for the sigma schedule.
    public let shift: Float

    public init(shift: Float = 1.0) {
        self.shift = shift
    }
}
