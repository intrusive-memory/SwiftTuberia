import CoreGraphics
@preconcurrency import MLX

@testable import Tuberia

/// Mock Renderer that produces a minimal valid CGImage for testing.
public final class MockRenderer: Renderer, @unchecked Sendable {
  public typealias Configuration = Void

  public var renderCallCount: Int = 0

  public required init(configuration: Void) {}

  // MARK: - Renderer

  public func render(_ input: DecodedOutput) throws -> RenderedOutput {
    renderCallCount += 1
    // Create a minimal 1x1 CGImage for testing
    let width = 1
    let height = 1
    let bitsPerComponent = 8
    let bytesPerRow = width * 4
    var pixelData: [UInt8] = [128, 128, 128, 255]

    guard
      let context = CGContext(
        data: &pixelData,
        width: width,
        height: height,
        bitsPerComponent: bitsPerComponent,
        bytesPerRow: bytesPerRow,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      ), let image = context.makeImage()
    else {
      throw PipelineError.renderingFailed(reason: "MockRenderer: failed to create CGImage")
    }

    return .image(image)
  }
}
