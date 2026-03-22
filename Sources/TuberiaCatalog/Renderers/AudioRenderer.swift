import Foundation
@preconcurrency import MLX
import Tuberia

/// Stateless renderer that converts decoded audio samples (MLXArray) into WAV Data.
///
/// Input: `DecodedOutput` with `.data` shape [B, samples] (float in -1.0 to 1.0 range)
///        and `AudioDecoderMetadata` providing `sampleRate`.
/// Output: `.audio(AudioData)` via `RenderedOutput`
///
/// When batch size B > 1, only the first audio track is rendered.
/// Produces standard PCM WAV format (16-bit, mono).
/// No model weights. No configuration. Freely concurrent.
public struct AudioRenderer: Renderer, Sendable {
    public typealias Configuration = Void

    public init(configuration: Void) {}

    public func render(_ input: DecodedOutput) throws -> RenderedOutput {
        let data = input.data
        let shape = data.shape

        // Validate input shape: [B, samples]
        guard shape.count == 2 else {
            throw PipelineError.renderingFailed(
                reason: "AudioRenderer expects [B, samples] input, got shape \(shape)"
            )
        }

        // Extract sample rate from metadata
        guard let audioMetadata = input.metadata as? AudioDecoderMetadata else {
            throw PipelineError.renderingFailed(
                reason: "AudioRenderer requires AudioDecoderMetadata, got \(type(of: input.metadata))"
            )
        }

        let sampleRate = audioMetadata.sampleRate
        let numSamples = shape[1]

        // Extract the first track from the batch
        let singleTrack = data[0]

        // Clamp float samples to [-1.0, 1.0] and convert to Int16
        let clamped = MLX.clip(singleTrack, min: -1.0, max: 1.0)
        let scaled = (clamped * Float(Int16.max)).asType(.int16)
        eval(scaled)

        let samples: [Int16] = scaled.asArray(Int16.self)

        guard samples.count == numSamples else {
            throw PipelineError.renderingFailed(
                reason: "AudioRenderer: sample count mismatch. Expected \(numSamples), got \(samples.count)"
            )
        }

        // Build WAV file
        let wavData = buildWAV(samples: samples, sampleRate: sampleRate, bitsPerSample: 16, numChannels: 1)

        return .audio(AudioData(data: wavData, sampleRate: sampleRate))
    }

    // MARK: - WAV Construction

    /// Build a valid WAV (RIFF) file from PCM samples.
    ///
    /// WAV file structure:
    /// - RIFF header (12 bytes)
    /// - fmt chunk (24 bytes)
    /// - data chunk header (8 bytes) + sample data
    private func buildWAV(samples: [Int16], sampleRate: Int, bitsPerSample: Int, numChannels: Int) -> Data {
        let bytesPerSample = bitsPerSample / 8
        let dataSize = UInt32(samples.count * bytesPerSample)
        let fmtChunkSize: UInt32 = 16
        let fileSize = 4 + (8 + fmtChunkSize) + (8 + dataSize) // "WAVE" + fmt chunk + data chunk

        var data = Data()
        data.reserveCapacity(Int(12 + 8 + fmtChunkSize + 8 + dataSize))

        // RIFF header
        data.append(contentsOf: [UInt8]("RIFF".utf8))
        data.append(littleEndian: fileSize)

        data.append(contentsOf: [UInt8]("WAVE".utf8))

        // fmt sub-chunk
        data.append(contentsOf: [UInt8]("fmt ".utf8))
        data.append(littleEndian: fmtChunkSize)
        data.append(littleEndian: UInt16(1))                                           // PCM format
        data.append(littleEndian: UInt16(numChannels))                                 // Mono
        data.append(littleEndian: UInt32(sampleRate))                                  // Sample rate
        data.append(littleEndian: UInt32(sampleRate * numChannels * bytesPerSample))   // Byte rate
        data.append(littleEndian: UInt16(numChannels * bytesPerSample))                // Block align
        data.append(littleEndian: UInt16(bitsPerSample))                               // Bits per sample

        // data sub-chunk
        data.append(contentsOf: [UInt8]("data".utf8))
        data.append(littleEndian: dataSize)

        // PCM sample data (little-endian Int16)
        for sample in samples {
            data.append(littleEndian: sample)
        }

        return data
    }
}

// MARK: - Data Extension for Little-Endian Writing

extension Data {
    fileprivate mutating func append<T: FixedWidthInteger>(littleEndian value: T) {
        var le = value.littleEndian
        Swift.withUnsafeBytes(of: &le) { buffer in
            self.append(contentsOf: buffer)
        }
    }
}
