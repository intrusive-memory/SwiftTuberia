import CoreGraphics
import Foundation

// MARK: - Renderer Output Types

public enum RenderedOutput: Sendable {
  case image(CGImage)
  case audio(AudioData)
  case video(VideoFrames)
}

public struct AudioData: Sendable {
  public let data: Data
  public let sampleRate: Int
  public init(data: Data, sampleRate: Int) {
    self.data = data
    self.sampleRate = sampleRate
  }
}

public struct VideoFrames: Sendable {
  public let frames: [CGImage]
  public let frameRate: Double
  public init(frames: [CGImage], frameRate: Double) {
    self.frames = frames
    self.frameRate = frameRate
  }
}

// MARK: - Renderer Protocol

public protocol Renderer: Sendable {
  /// Configuration type (`Void` for stateless renderers like ImageRenderer).
  associatedtype Configuration: Sendable

  /// Construct the renderer from its configuration. The pipeline calls this
  /// during `DiffusionPipeline.init(recipe:)` to instantiate the component.
  init(configuration: Configuration)

  /// Render decoded output into the final format.
  func render(_ input: DecodedOutput) throws -> RenderedOutput
}
