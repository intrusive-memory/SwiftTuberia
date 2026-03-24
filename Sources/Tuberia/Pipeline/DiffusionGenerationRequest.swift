import CoreGraphics

/// Request parameters for a diffusion generation run.
public struct DiffusionGenerationRequest: Sendable {
    public let prompt: String
    public let negativePrompt: String?
    public let width: Int
    public let height: Int
    public let steps: Int
    public let guidanceScale: Float
    public let seed: UInt32?
    public let loRA: LoRAConfig?
    public let referenceImages: [CGImage]?
    public let strength: Float?

    public init(prompt: String, negativePrompt: String? = nil,
                width: Int, height: Int, steps: Int, guidanceScale: Float,
                seed: UInt32? = nil, loRA: LoRAConfig? = nil,
                referenceImages: [CGImage]? = nil, strength: Float? = nil) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.seed = seed
        self.loRA = loRA
        self.referenceImages = referenceImages
        self.strength = strength
    }
}
