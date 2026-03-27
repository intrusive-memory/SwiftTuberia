import CoreGraphics
import Foundation
import Testing

@testable import Tuberia

// MARK: - CGImageToMLXArray Tests
//
// NOTE: These tests validate CGImage creation and basic structure.
// The full conversion tests (testing actual MLXArray values from cgImageToMLXArray)
// require access to the private DiffusionPipeline method — those tests are integration
// tests that will be implemented when the full pipeline can be instantiated.

@Suite("CGImageToMLXArray Tests", .serialized)
struct CGImageToMLXArrayTests {

    /// Helper to create a synthetic CGImage with known pixel values.
    /// Uses CGDataProvider directly (not CGContext) so it works in headless CI environments.
    private func createSyntheticCGImage(
        width: Int,
        height: Int,
        hasAlpha: Bool
    ) -> CGImage? {
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }

        // Always use 4 bytes per pixel (RGBA) — CGImage with 3 bpp is poorly supported
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var pixelData = [UInt8](repeating: 0, count: height * width * bytesPerPixel)

        // Fill with a pattern:
        // Red quadrant (top-left): R=255, G=0, B=0
        // Green quadrant (top-right): R=0, G=255, B=0
        // Blue quadrant (bottom-left): R=0, G=0, B=255
        // White quadrant (bottom-right): R=255, G=255, B=255
        for y in 0..<height {
            for x in 0..<width {
                let index = (y * width + x) * bytesPerPixel
                let isTopHalf = y < height / 2
                let isLeftHalf = x < width / 2

                if isTopHalf && isLeftHalf {
                    pixelData[index] = 255; pixelData[index + 1] = 0; pixelData[index + 2] = 0
                } else if isTopHalf && !isLeftHalf {
                    pixelData[index] = 0; pixelData[index + 1] = 255; pixelData[index + 2] = 0
                } else if !isTopHalf && isLeftHalf {
                    pixelData[index] = 0; pixelData[index + 1] = 0; pixelData[index + 2] = 255
                } else {
                    pixelData[index] = 255; pixelData[index + 1] = 255; pixelData[index + 2] = 255
                }

                // Alpha channel: fully opaque if hasAlpha, otherwise stored but marked as skip
                pixelData[index + 3] = 255
            }
        }

        let data = Data(pixelData)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }

        let bitmapInfo: CGBitmapInfo = hasAlpha
            ? CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
            : CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)

        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }

    // MARK: - Test Cases

    @Test("CGImage creation: synthetic RGB image has correct dimensions")
    func testSyntheticRGBImageDimensions() {
        let image = createSyntheticCGImage(width: 10, height: 10, hasAlpha: false)
        #expect(image != nil)
        #expect(image?.width == 10)
        #expect(image?.height == 10)
    }

    @Test("CGImage creation: synthetic RGBA image has correct dimensions")
    func testSyntheticRGBAImageDimensions() {
        let image = createSyntheticCGImage(width: 64, height: 64, hasAlpha: true)
        #expect(image != nil)
        #expect(image?.width == 64)
        #expect(image?.height == 64)
    }

    @Test("CGImage creation: various sizes create successfully")
    func testVariousImageSizes() {
        let sizes = [(32, 32), (64, 128), (256, 128), (512, 512)]
        for (width, height) in sizes {
            let image = createSyntheticCGImage(width: width, height: height, hasAlpha: true)
            #expect(image != nil, "Failed to create image at \(width)×\(height)")
            #expect(image?.width == width)
            #expect(image?.height == height)
        }
    }

    @Test("DiffusionGenerationRequest: creates with referenceImages parameter")
    func testRequestCreationWithReferenceImages() throws {
        let image = try #require(createSyntheticCGImage(width: 10, height: 10, hasAlpha: false))
        let request = DiffusionGenerationRequest(
            prompt: "test",
            width: 512,
            height: 512,
            steps: 1,
            guidanceScale: 1.0,
            seed: 42,
            referenceImages: [image],
            strength: 0.8
        )
        #expect(request.height == 512)
        #expect(request.width == 512)
        #expect(request.referenceImages?.count == 1)
    }
}
