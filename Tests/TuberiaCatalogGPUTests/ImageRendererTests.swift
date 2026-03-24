import Testing
import CoreGraphics
@preconcurrency import MLX
@testable import TuberiaCatalog
import Tuberia

@Suite("ImageRenderer Tests")
struct ImageRendererTests {

    @Test("2x2 pixel array produces CGImage with correct dimensions")
    func correctDimensions() throws {
        let renderer = ImageRenderer(configuration: ())

        let pixelValues: [Float] = [
            1.0, 0.0, 0.0,   0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,   1.0, 1.0, 1.0
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

    @Test("Known pixel values produce correct RGB bytes")
    func correctPixelValues() throws {
        let renderer = ImageRenderer(configuration: ())

        let pixelValues: [Float] = [1.0, 0.0, 0.0]
        let pixelData = MLXArray(pixelValues).reshaped([1, 1, 1, 3])
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: pixelData, metadata: metadata)

        let output = try renderer.render(input)

        guard case .image(let cgImage) = output,
              let dataProvider = cgImage.dataProvider,
              let data = dataProvider.data,
              let bytes = CFDataGetBytePtr(data) else {
            Issue.record("Could not extract pixel data from CGImage")
            return
        }

        #expect(bytes[0] == 255) // R
        #expect(bytes[1] == 0)   // G
        #expect(bytes[2] == 0)   // B
    }

    @Test("Batch input renders first image only")
    func batchRendersFirstImage() throws {
        let renderer = ImageRenderer(configuration: ())

        let batchData = MLXArray.zeros([2, 3, 3, 3]) + 0.5
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: batchData, metadata: metadata)

        let output = try renderer.render(input)

        guard case .image(let cgImage) = output else {
            Issue.record("Expected .image output")
            return
        }

        #expect(cgImage.width == 3)
        #expect(cgImage.height == 3)
    }

    @Test("Out-of-range values are clamped to [0, 255]")
    func clampingBehavior() throws {
        let renderer = ImageRenderer(configuration: ())

        let pixelValues: [Float] = [-0.5, 1.5, 0.5]
        let pixelData = MLXArray(pixelValues).reshaped([1, 1, 1, 3])
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: pixelData, metadata: metadata)

        let output = try renderer.render(input)

        guard case .image(let cgImage) = output,
              let dataProvider = cgImage.dataProvider,
              let data = dataProvider.data,
              let bytes = CFDataGetBytePtr(data) else {
            Issue.record("Could not extract pixel data")
            return
        }

        #expect(bytes[0] == 0)   // -0.5 clamped to 0
        #expect(bytes[1] == 255) // 1.5 clamped to 255
        #expect(bytes[2] == 127 || bytes[2] == 128)
    }

    // MARK: - Error Paths

    @Test("3D input (missing batch dim) throws renderingFailed")
    func invalidShapeThrows() throws {
        let renderer = ImageRenderer(configuration: ())

        let badData = MLXArray.zeros([2, 2, 3])
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: badData, metadata: metadata)

        #expect(throws: PipelineError.self) {
            try renderer.render(input)
        }
    }

    @Test("Wrong channel count (4 instead of 3) throws renderingFailed")
    func wrongChannelsThrows() throws {
        let renderer = ImageRenderer(configuration: ())

        let badData = MLXArray.zeros([1, 2, 2, 4])
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: badData, metadata: metadata)

        #expect(throws: PipelineError.self) {
            try renderer.render(input)
        }
    }
}
