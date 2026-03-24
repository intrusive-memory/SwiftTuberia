import Testing
import Foundation
@preconcurrency import MLX
@testable import TuberiaCatalog
import Tuberia

@Suite("AudioRenderer Tests")
struct AudioRendererTests {

    @Test("WAV header has correct RIFF/WAVE/fmt/data structure")
    func correctWAVHeader() throws {
        let renderer = AudioRenderer(configuration: ())

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

        #expect(String(data: wavData[0..<4], encoding: .ascii) == "RIFF")
        #expect(String(data: wavData[8..<12], encoding: .ascii) == "WAVE")
        #expect(String(data: wavData[12..<16], encoding: .ascii) == "fmt ")

        let format = wavData[20..<22].withUnsafeBytes { $0.load(as: UInt16.self).littleEndian }
        #expect(format == 1) // PCM

        let channels = wavData[22..<24].withUnsafeBytes { $0.load(as: UInt16.self).littleEndian }
        #expect(channels == 1) // Mono

        #expect(String(data: wavData[36..<40], encoding: .ascii) == "data")
    }

    @Test("Sample rate from metadata is written into WAV header")
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

        let headerSampleRate = audio.data[24..<28].withUnsafeBytes {
            $0.load(as: UInt32.self).littleEndian
        }
        #expect(headerSampleRate == 22050)
    }

    @Test("WAV file size matches 44-byte header + 16-bit samples")
    func validWAVFormat() throws {
        let renderer = AudioRenderer(configuration: ())

        let numSamples = 44100
        let sampleValues: [Float] = (0..<numSamples).map { i in
            sin(Float(i) * 2.0 * Float.pi * 440.0 / 44100.0)
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

        #expect(wavData.count == 44 + numSamples * 2)

        let dataSize = wavData[40..<44].withUnsafeBytes {
            $0.load(as: UInt32.self).littleEndian
        }
        #expect(dataSize == UInt32(numSamples * 2))

        let bitsPerSample = wavData[34..<36].withUnsafeBytes {
            $0.load(as: UInt16.self).littleEndian
        }
        #expect(bitsPerSample == 16)
    }

    // MARK: - Error Paths

    @Test("Non-AudioDecoderMetadata throws renderingFailed")
    func wrongMetadataThrows() throws {
        let renderer = AudioRenderer(configuration: ())

        let audioData = MLXArray.zeros([1, 100])
        let metadata = ImageDecoderMetadata(scalingFactor: 1.0)
        let input = DecodedOutput(data: audioData, metadata: metadata)

        #expect(throws: PipelineError.self) {
            try renderer.render(input)
        }
    }

    @Test("3D input instead of 2D throws renderingFailed")
    func invalidShapeThrows() throws {
        let renderer = AudioRenderer(configuration: ())

        let badData = MLXArray.zeros([1, 100, 2])
        let metadata = AudioDecoderMetadata(scalingFactor: 1.0, sampleRate: 44100)
        let input = DecodedOutput(data: badData, metadata: metadata)

        #expect(throws: PipelineError.self) {
            try renderer.render(input)
        }
    }

    @Test("Extreme float values are clamped, not rejected")
    func clampingSamples() throws {
        let renderer = AudioRenderer(configuration: ())

        let sampleValues: [Float] = [-2.0, 0.0, 2.0]
        let audioData = MLXArray(sampleValues).reshaped([1, 3])
        let metadata = AudioDecoderMetadata(scalingFactor: 1.0, sampleRate: 44100)
        let input = DecodedOutput(data: audioData, metadata: metadata)

        let output = try renderer.render(input)

        guard case .audio(let audio) = output else {
            Issue.record("Expected .audio output")
            return
        }

        #expect(audio.data.count == 44 + 6)
    }
}
