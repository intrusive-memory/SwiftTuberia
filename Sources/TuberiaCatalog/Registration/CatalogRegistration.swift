import Foundation
import SwiftAcervo
import Tuberia

/// Describes a catalog component that can be downloaded and used by the pipeline.
///
/// This is the TuberiaCatalog-local component descriptor type. When SwiftAcervo v2
/// ships its own `ComponentDescriptor` and registry API, this type will be migrated
/// to use those. For now, this provides the metadata needed for component discovery
/// and download orchestration.
public struct ComponentDescriptor: Sendable {
    /// Unique component identifier (e.g., "t5-xxl-encoder-int4").
    public let componentId: String
    /// The role this component plays in the pipeline.
    public let componentType: ComponentType
    /// HuggingFace repository (org/repo format) where the weights are hosted.
    public let huggingFaceRepo: String
    /// Glob patterns for the files needed from the repository.
    public let filePatterns: [String]
    /// Estimated total size in bytes for memory budgeting.
    public let estimatedSizeBytes: Int
    /// Optional SHA-256 checksums for integrity verification.
    /// `nil` means skip verification (checksums populated after weight conversion).
    public let sha256Checksums: [String: String]?

    public init(
        componentId: String,
        componentType: ComponentType,
        huggingFaceRepo: String,
        filePatterns: [String],
        estimatedSizeBytes: Int,
        sha256Checksums: [String: String]? = nil
    ) {
        self.componentId = componentId
        self.componentType = componentType
        self.huggingFaceRepo = huggingFaceRepo
        self.filePatterns = filePatterns
        self.estimatedSizeBytes = estimatedSizeBytes
        self.sha256Checksums = sha256Checksums
    }
}

/// The role a component plays in the pipeline.
public enum ComponentType: String, Sendable {
    case encoder
    case scheduler
    case backbone
    case decoder
    case renderer
}

/// Registers and tracks all TuberiaCatalog's shared component descriptors.
///
/// Registration is triggered by calling `CatalogRegistration.ensureRegistered()`.
/// This is idempotent -- duplicate registration is silently ignored.
///
/// Only weighted components are registered (T5-XXL encoder, SDXL VAE decoder).
/// Schedulers and renderers have no weights and need no component descriptor.
///
/// Components can be queried by ID or enumerated. This acts as a lightweight
/// registry until SwiftAcervo v2's full Component Registry is available.
public final class CatalogRegistration: @unchecked Sendable {

    /// Singleton instance.
    public static let shared = CatalogRegistration()

    /// Thread-safe storage for registered descriptors.
    private var descriptors: [String: ComponentDescriptor] = [:]
    private let lock = NSLock()
    private var isRegistered = false

    private init() {}

    // MARK: - Component Descriptors

    /// T5-XXL Encoder (int4 quantized) component descriptor.
    public static let t5XXLEncoderDescriptor = ComponentDescriptor(
        componentId: "t5-xxl-encoder-int4",
        componentType: .encoder,
        huggingFaceRepo: "intrusive-memory/t5-xxl-int4-mlx",
        filePatterns: ["*.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json"],
        estimatedSizeBytes: 1_288_490_188, // ~1.2 GB
        sha256Checksums: nil // Populated after weight conversion
    )

    /// SDXL VAE Decoder (fp16) component descriptor.
    public static let sdxlVAEDecoderDescriptor = ComponentDescriptor(
        componentId: "sdxl-vae-decoder-fp16",
        componentType: .decoder,
        huggingFaceRepo: "intrusive-memory/sdxl-vae-fp16-mlx",
        filePatterns: ["*.safetensors", "config.json"],
        estimatedSizeBytes: 167_772_160, // ~160 MB
        sha256Checksums: nil // Populated after weight conversion
    )

    // MARK: - Registration

    /// Ensure all catalog components are registered.
    /// Safe to call multiple times -- only performs registration on first call.
    public func ensureRegistered() {
        lock.lock()
        defer { lock.unlock() }

        guard !isRegistered else { return }

        register(CatalogRegistration.t5XXLEncoderDescriptor)
        register(CatalogRegistration.sdxlVAEDecoderDescriptor)

        isRegistered = true
    }

    /// Register a component descriptor. Duplicate IDs are silently ignored.
    public func register(_ descriptor: ComponentDescriptor) {
        lock.lock()
        defer { lock.unlock() }

        // Deduplicate: same ID = no-op
        if descriptors[descriptor.componentId] != nil {
            return
        }
        descriptors[descriptor.componentId] = descriptor
    }

    // MARK: - Queries

    /// Look up a component descriptor by its ID.
    public func descriptor(for componentId: String) -> ComponentDescriptor? {
        lock.lock()
        defer { lock.unlock() }
        return descriptors[componentId]
    }

    /// Return all registered component IDs.
    public func registeredComponentIds() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return Array(descriptors.keys)
    }

    /// Return all registered component descriptors.
    public func registeredDescriptors() -> [ComponentDescriptor] {
        lock.lock()
        defer { lock.unlock() }
        return Array(descriptors.values)
    }

    /// Check if a component is registered.
    public func isComponentRegistered(_ componentId: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return descriptors[componentId] != nil
    }

    /// Get the HuggingFace repo for a component. Used to bridge to Acervo v1's
    /// download API (`Acervo.ensureAvailable(modelId:)`).
    public func huggingFaceRepo(for componentId: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return descriptors[componentId]?.huggingFaceRepo
    }

    // MARK: - Reset (for testing)

    /// Reset the registry to empty state. Only for testing.
    internal func reset() {
        lock.lock()
        defer { lock.unlock() }
        descriptors.removeAll()
        isRegistered = false
    }
}
