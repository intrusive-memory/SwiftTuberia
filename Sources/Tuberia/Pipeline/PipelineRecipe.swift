/// Identifies a pipe segment role in the pipeline.
public enum PipelineRole: String, Sendable, CaseIterable {
    case encoder, scheduler, backbone, decoder, renderer
}

/// Strategy for computing unconditional embeddings used in classifier-free guidance.
public enum UnconditionalEmbeddingStrategy: Sendable {
    /// Encode "" through the same TextEncoder (PixArt, SD, SDXL).
    case emptyPrompt
    /// All-zero embedding of the given shape.
    case zeroVector(shape: [Int])
    /// Skip CFG entirely -- guidance scale is informational only (FLUX).
    case none
}

/// Declares which pipe segments to connect and how to configure them.
///
/// Model plugins provide recipes. The DiffusionPipeline instantiates
/// components from their configurations and validates shape contracts
/// at assembly time.
public protocol PipelineRecipe: Sendable {
    associatedtype Encoder: TextEncoder
    associatedtype Sched: Scheduler
    associatedtype Back: Backbone
    associatedtype Dec: Decoder
    associatedtype Rend: Renderer

    var encoderConfig: Encoder.Configuration { get }
    var schedulerConfig: Sched.Configuration { get }
    var backboneConfig: Back.Configuration { get }
    var decoderConfig: Dec.Configuration { get }
    var rendererConfig: Rend.Configuration { get }

    var supportsImageToImage: Bool { get }
    var unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy { get }
    var allComponentIds: [String] { get }

    func quantizationFor(_ role: PipelineRole) -> QuantizationConfig
    func validate() throws
}
