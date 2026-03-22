import Testing
import Foundation
@preconcurrency import MLX
@testable import TuberiaCatalog
import Tuberia

@Suite("AudioRenderer Tests")
struct AudioRendererTests {

    @Test("AudioRenderer conforms to Renderer with Configuration = Void")
    func protocolConformance() {
        let renderer = AudioRenderer(configuration: ())
        _ = renderer
    }

    @Test("Known sample array produces WAV data with correct header")
    func correctWAVHeader() throws {
        let renderer = AudioRenderer(configuration: ())

        // Create a simple audio signal: [1, 100] samples
        let sampleValues: [Float] = (0..<100).map { i in
            sin(Float(i) * 0.1) * 0.5
        }
        let audioData = MLXArray(sampleValues).reshaped([1, 100])
        let metadata = AudioDecoderMetadata(scalingFactor: 1.0, sampleRate: 44100)
        let input = DecodedOutput(data: audioData, metadata: metadata)

        let output = try renderer.render(input)

        guard case .audio(let audio) = output else {
            Issue.record("Expected .audio output, got \(output)")
            return
        }

        let wavData = audio.data

        // Verify WAV header
        // RIFF magic
        let riff = String(data: wavData[0..<4], encoding: .ascii)
        #expect(riff == "RIFF")

        // WAVE magic
        let wave = String(data: wavData[8..<12], encoding: .ascii)
        #expect(wave == "WAVE")

        // fmt chunk
        let fmt = String(data: wavData[12..<16], encoding: .ascii)
        #expect(fmt == "fmt ")

        // PCM format (1)
        let format = wavData[20..<22].withUnsafeBytes { $0.load(as: UInt16.self).littleEndian }
        #expect(format == 1) // PCM

        // Mono (1 channel)
        let channels = wavData[22..<24].withUnsafeBytes { $0.load(as: UInt16.self).littleEndian }
        #expect(channels == 1)

        // data chunk
        let dataChunk = String(data: wavData[36..<40], encoding: .ascii)
        #expect(dataChunk == "data")
    }

    @Test("Correct sample rate from metadata is reflected in WAV header")
    func sampleRateInHeader() throws {
        let renderer = AudioRenderer(configuration: ())

        let audioData = MLXArray.zeros([1, 10])
        let metadata = AudioDecoderMetadata(scalingFactor: 1.0, sampleRate: 22050)
        let input = DecodedOutput(data: audioData, metadata: metadata)

        let output = try renderer.render(input)

        guard case .audio(let audio) = output else {
            Issue.record("Expected .audio output")
            return
        }

        #expect(audio.sampleRate == 22050)

        // Check sample rate in WAV header (bytes 24-28)
        let wavData = audio.data
        let headerSampleRate = wavData[24..<28].withUnsafeBytes {
            $0.load(as: UInt32.self).littleEndian
        }
        #expect(headerSampleRate == 22050)
    }

    @Test("Output Data is a valid WAV format")
    func validWAVFormat() throws {
        let renderer = AudioRenderer(configuration: ())

        // 1 second at 44100 Hz
        let numSamples = 44100
        let sampleValues: [Float] = (0..<numSamples).map { i in
            sin(Float(i) * 2.0 * Float.pi * 440.0 / 44100.0) // 440 Hz sine
        }
        let audioData = MLXArray(sampleValues).reshaped([1, numSamples])
        let metadata = AudioDecoderMetadata(scalingFactor: 1.0, sampleRate: 44100)
        let input = DecodedOutput(data: audioData, metadata: metadata)

        let output = try renderer.render(input)

        guard case .audio(let audio) = output else {
            Issue.record("Expected .audio output")
            return
        }

        let wavData = audio.data

        // WAV file size should be: 44 byte header + 2 bytes per sample * numSamples
        let expectedSize = 44 + numSamples * 2
        #expect(wavData.count == expectedSize)

        // Verify data size in header (bytes 40-44)
        let dataSize = wavData[40..<44].withUnsafeBytes {
            $0.load(as: UInt32.self).littleEndian
        }
        #expect(dataSize == UInt32(numSamples * 2))

        // Bits per sample (bytes 34-36)
        let bitsPerSample = wavData[34..<36].withUnsafeBytes {
            $0.load(as: UInt16.self).littleEndian
        }
        #expect(bitsPerSample == 16)
    }

    @Test("Non-AudioDecoderMetadata throws renderingFailed")
    func wrongMetadataThrows() throws {
        let renderer = AudioRenderer(configuration: ())

        let audioData = MLXArray.zeros([1, 100])
        // Use image metadata instead of audio metadata
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: audioData, metadata: metadata)

        #expect(throws: PipelineError.self) {
            try renderer.render(input)
        }
    }

    @Test("Invalid input shape throws renderingFailed")
    func invalidShapeThrows() throws {
        let renderer = AudioRenderer(configuration: ())

        // 3D input instead of 2D
        let badData = MLXArray.zeros([1, 100, 2])
        let metadata = AudioDecoderMetadata(scalingFactor: 1.0, sampleRate: 44100)
        let input = DecodedOutput(data: badData, metadata: metadata)

        #expect(throws: PipelineError.self) {
            try renderer.render(input)
        }
    }

    @Test("Float samples are clamped to [-1, 1] range before conversion")
    func clampingSamples() throws {
        let renderer = AudioRenderer(configuration: ())

        // Extreme values that should be clamped
        let sampleValues: [Float] = [-2.0, 0.0, 2.0]
        let audioData = MLXArray(sampleValues).reshaped([1, 3])
        let metadata = AudioDecoderMetadata(scalingFactor: 1.0, sampleRate: 44100)
        let input = DecodedOutput(data: audioData, metadata: metadata)

        // Should not throw -- extreme values are clamped
        let output = try renderer.render(input)

        guard case .audio(let audio) = output else {
            Issue.record("Expected .audio output")
            return
        }

        // Verify the data was produced (header + 3 samples * 2 bytes)
        #expect(audio.data.count == 44 + 6)
    }
}
