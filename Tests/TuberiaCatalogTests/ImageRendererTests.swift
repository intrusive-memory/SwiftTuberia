import Testing
import CoreGraphics
@preconcurrency import MLX
@testable import TuberiaCatalog
import Tuberia

@Suite("ImageRenderer Tests")
struct ImageRendererTests {

    @Test("ImageRenderer conforms to Renderer with Configuration = Void")
    func protocolConformance() {
        let renderer = ImageRenderer(configuration: ())
        // If this compiles and creates, the protocol conformance is verified.
        _ = renderer
    }

    @Test("Known 2x2 pixel array produces CGImage with correct dimensions")
    func correctDimensions() throws {
        let renderer = ImageRenderer(configuration: ())

        // Create a 2x2 image: [1, 2, 2, 3]
        let pixelValues: [Float] = [
            // Row 0
            1.0, 0.0, 0.0,   // Red pixel
            0.0, 1.0, 0.0,   // Green pixel
            // Row 1
            0.0, 0.0, 1.0,   // Blue pixel
            1.0, 1.0, 1.0    // White pixel
        ]
        let pixelData = MLXArray(pixelValues).reshaped([1, 2, 2, 3])
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: pixelData, metadata: metadata)

        let output = try renderer.render(input)

        guard case .image(let cgImage) = output else {
            Issue.record("Expected .image output, got \(output)")
            return
        }

        #expect(cgImage.width == 2)
        #expect(cgImage.height == 2)
    }

    @Test("Known pixel values are correctly represented in output CGImage")
    func correctPixelValues() throws {
        let renderer = ImageRenderer(configuration: ())

        // Single solid red pixel: [1, 1, 1, 3]
        let pixelValues: [Float] = [1.0, 0.0, 0.0]
        let pixelData = MLXArray(pixelValues).reshaped([1, 1, 1, 3])
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: pixelData, metadata: metadata)

        let output = try renderer.render(input)

        guard case .image(let cgImage) = output else {
            Issue.record("Expected .image output")
            return
        }

        // Extract pixel data from CGImage
        guard let dataProvider = cgImage.dataProvider,
              let data = dataProvider.data,
              let bytes = CFDataGetBytePtr(data) else {
            Issue.record("Could not extract pixel data from CGImage")
            return
        }

        // RGBA format: red pixel should be [255, 0, 0, ...]
        #expect(bytes[0] == 255) // R
        #expect(bytes[1] == 0)   // G
        #expect(bytes[2] == 0)   // B
    }

    @Test("Batch dimension (B > 1) renders first image only")
    func batchRendersFirstImage() throws {
        let renderer = ImageRenderer(configuration: ())

        // Batch of 2 images: [2, 3, 3, 3]
        let batchData = MLXArray.zeros([2, 3, 3, 3]) + 0.5
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: batchData, metadata: metadata)

        let output = try renderer.render(input)

        guard case .image(let cgImage) = output else {
            Issue.record("Expected .image output")
            return
        }

        // Should be 3x3 (first image from batch)
        #expect(cgImage.width == 3)
        #expect(cgImage.height == 3)
    }

    @Test("Values outside 0-1 range are clamped")
    func clampingBehavior() throws {
        let renderer = ImageRenderer(configuration: ())

        // Values outside [0, 1] should be clamped
        let pixelValues: [Float] = [-0.5, 1.5, 0.5]
        let pixelData = MLXArray(pixelValues).reshaped([1, 1, 1, 3])
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: pixelData, metadata: metadata)

        let output = try renderer.render(input)

        guard case .image(let cgImage) = output else {
            Issue.record("Expected .image output")
            return
        }

        guard let dataProvider = cgImage.dataProvider,
              let data = dataProvider.data,
              let bytes = CFDataGetBytePtr(data) else {
            Issue.record("Could not extract pixel data")
            return
        }

        #expect(bytes[0] == 0)   // -0.5 clamped to 0.0 -> 0
        #expect(bytes[1] == 255) // 1.5 clamped to 1.0 -> 255
        // 0.5 * 255 = 127.5, truncates to 127 or rounds to 128 depending on implementation
        #expect(bytes[2] == 127 || bytes[2] == 128, "0.5 should map to 127 or 128")
    }

    @Test("Invalid input shape throws renderingFailed")
    func invalidShapeThrows() throws {
        let renderer = ImageRenderer(configuration: ())

        // 3D input instead of 4D
        let badData = MLXArray.zeros([2, 2, 3])
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: badData, metadata: metadata)

        #expect(throws: PipelineError.self) {
            try renderer.render(input)
        }
    }

    @Test("Wrong channel count throws renderingFailed")
    func wrongChannelsThrows() throws {
        let renderer = ImageRenderer(configuration: ())

        // 4 channels instead of 3
        let badData = MLXArray.zeros([1, 2, 2, 4])
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: badData, metadata: metadata)

        #expect(throws: PipelineError.self) {
            try renderer.render(input)
        }
    }
}
