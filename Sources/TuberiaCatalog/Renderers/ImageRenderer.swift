import CoreGraphics
import Foundation
@preconcurrency import MLX
import Tuberia

/// Stateless renderer that converts decoded pixel data (MLXArray) into CGImage.
///
/// Input: `DecodedOutput` with `.data` shape [B, H, W, 3] (float in 0.0-1.0 range)
/// Output: `.image(CGImage)` via `RenderedOutput`
///
/// When batch size B > 1, only the first image is rendered.
/// No model weights. No configuration. Freely concurrent.
public struct ImageRenderer: Renderer, Sendable {
  public typealias Configuration = Void

  public init(configuration: Void) {}

  public func render(_ input: DecodedOutput) throws -> RenderedOutput {
    let data = input.data
    let shape = data.shape

    // Validate input shape: [B, H, W, 3]
    guard shape.count == 4, shape[3] == 3 else {
      throw PipelineError.renderingFailed(
        reason: "ImageRenderer expects [B, H, W, 3] input, got shape \(shape)"
      )
    }

    let height = shape[1]
    let width = shape[2]

    // Extract the first image from the batch
    let singleImage = data[0]

    // Flatten to contiguous array and convert float [0,1] -> UInt8 [0,255]
    // Clamp values to [0, 1] range
    let clamped = MLX.clip(singleImage, min: 0.0, max: 1.0)
    let scaled = (clamped * 255.0).asType(.uint8)

    // Evaluate to ensure computation completes
    eval(scaled)

    // Extract pixel bytes
    let totalBytes = height * width * 3
    let pixelBytes: [UInt8] = scaled.asArray(UInt8.self)

    guard pixelBytes.count == totalBytes else {
      throw PipelineError.renderingFailed(
        reason:
          "ImageRenderer: pixel byte count mismatch. Expected \(totalBytes), got \(pixelBytes.count)"
      )
    }

    // Convert RGB to RGBA for CGContext (add alpha=255)
    var rgbaBytes = [UInt8](repeating: 255, count: height * width * 4)
    for i in 0..<(height * width) {
      rgbaBytes[i * 4 + 0] = pixelBytes[i * 3 + 0]
      rgbaBytes[i * 4 + 1] = pixelBytes[i * 3 + 1]
      rgbaBytes[i * 4 + 2] = pixelBytes[i * 3 + 2]
      // Alpha already set to 255
    }

    // Create CGImage from RGBA data
    let bitsPerComponent = 8
    let bytesPerRow = width * 4
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)

    guard
      let context = CGContext(
        data: &rgbaBytes,
        width: width,
        height: height,
        bitsPerComponent: bitsPerComponent,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: bitmapInfo.rawValue
      )
    else {
      throw PipelineError.renderingFailed(
        reason: "ImageRenderer: failed to create CGContext"
      )
    }

    guard let cgImage = context.makeImage() else {
      throw PipelineError.renderingFailed(
        reason: "ImageRenderer: failed to create CGImage from context"
      )
    }

    return .image(cgImage)
  }
}
