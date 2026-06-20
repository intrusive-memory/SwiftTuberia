import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
import Testing
import Tuberia

@testable import TuberiaCatalog

/// Real-weights validation for the #45 VAE-decode fixes: feathered tiled decode
/// (seam parity vs. full-frame) and fp16 decode (parity vs. fp32, NaN-free).
///
/// These load the actual `intrusive-memory/sdxl-vae-fp16-mlx` checkpoint from the
/// shared models container and run real decodes on the GPU through the public
/// `SDXLVAEDecoder.decode` path, so they only run when the weights are present
/// locally (and are skipped — not failed — in CI runners that don't have them).
/// Set `SDXL_VAE_WEIGHTS` to override the path.
@Suite("SDXLVAEDecoder real-weights decode (#45)", .serialized)
struct SDXLVAEDecoderRealWeightsTests {

  // MARK: - Helpers

  /// Resolves the SDXL VAE safetensors from the first readable candidate:
  /// `$SDXL_VAE_WEIGHTS`, a `/tmp` staging copy (xctest is sandboxed and cannot
  /// open files under `~/Library/Group Containers`, so weights must be staged to
  /// an accessible path — see the repo's test-model linking), then the shared
  /// models container as a best-effort fallback. Returns nil if none are
  /// readable (the test then self-skips). Stage with, e.g.:
  ///   mkdir -p /tmp/sdxl-vae-fp16-mlx && ln -f \
  ///     "$HOME/Library/Group Containers/group.intrusive-memory.models/SharedModels/\
  /// intrusive-memory_sdxl-vae-fp16-mlx/model.safetensors" /tmp/sdxl-vae-fp16-mlx/
  private static func weightsURL() -> URL? {
    let fm = FileManager.default
    var candidates: [URL] = []
    if let override = ProcessInfo.processInfo.environment["SDXL_VAE_WEIGHTS"] {
      candidates.append(URL(fileURLWithPath: override))
    }
    candidates.append(URL(fileURLWithPath: "/tmp/sdxl-vae-fp16-mlx/model.safetensors"))
    candidates.append(
      fm.homeDirectoryForCurrentUser.appendingPathComponent(
        "Library/Group Containers/group.intrusive-memory.models/SharedModels"
          + "/intrusive-memory_sdxl-vae-fp16-mlx/model.safetensors"))
    // Probe by actually opening the file: under the sandbox `fileExists` can be
    // true for a path that cannot be opened, so a stat-only check is unreliable.
    return candidates.first { (try? Data(contentsOf: $0, options: .alwaysMapped)) != nil }
  }

  /// Builds a decoder with `config` and applies the checkpoint at `url` through
  /// the decoder's real key mapping.
  private static func loadDecoder(
    _ url: URL, config: SDXLVAEDecoderConfiguration
  ) throws -> SDXLVAEDecoder {
    let decoder = try SDXLVAEDecoder(configuration: config)
    let raw = try loadArrays(url: url)
    let mapper = decoder.keyMapping
    var mapped: [String: MLXArray] = [:]
    for (k, v) in raw {
      if let dest = mapper(k) { mapped[dest] = v }
    }
    try decoder.apply(weights: ModuleParameters(parameters: mapped))
    return decoder
  }

  /// PSNR over two [0,1] pixel tensors (peak signal = 1.0).
  private static func psnrDB(_ a: MLXArray, _ b: MLXArray) -> Float {
    let mse = MLX.mean(MLX.square(a - b)).item(Float.self)
    if mse <= 1e-12 { return Float.infinity }
    return 10.0 * log10(1.0 / mse)
  }

  // MARK: - Tests

  /// Feathered tiled decode must reproduce the full-frame decode on real weights
  /// (high PSNR) and must NOT introduce a discontinuity at the tile boundaries
  /// (boundary gradient stays close to the interior gradient — no seam).
  @Test("tiled decode is seam-free vs full-frame on real weights")
  func tiledSeamFreeOnRealWeights() throws {
    guard let url = Self.weightsURL() else {
      Issue.record("SDXL VAE weights not found; skipping real-weights tiled-decode test")
      return
    }
    let full = try Self.loadDecoder(url, config: .init(decodeDType: .float32))
    let tiled = try Self.loadDecoder(
      url,
      config: .init(decodeDType: .float32, decodeTileLatentSize: 32, decodeTileLatentOverlap: 8))

    // 96×96 latent → 768×768 px, tile 32 (stride 256 px) → interior seams at
    // pixel rows/cols 256 and 512. The decoder scales the latent internally.
    MLXRandom.seed(7)
    let latent = MLXRandom.normal([1, 96, 96, 4])
    eval(latent)

    let outFull = try full.decode(latent).data
    let outTiled = try tiled.decode(latent).data
    eval(outFull, outTiled)

    #expect(outTiled.shape == outFull.shape)
    #expect(outTiled.shape == [1, 768, 768, 3])
    #expect(!MLX.any(MLX.isNaN(outTiled)).item(Bool.self), "tiled decode produced NaNs")

    let psnr = Self.psnrDB(outFull, outTiled)
    print("[#45] tiled-vs-full PSNR = \(psnr) dB")
    #expect(psnr > 30.0, "tiled decode diverges from full-frame (PSNR \(psnr) dB)")

    // Seam check: mean |Δ| across the tile-boundary rows vs the mean |Δ| across
    // all adjacent rows. A hard-crop stitch spikes this ratio; a feathered blend
    // keeps it ~1.
    let img = outTiled  // [1,768,768,3] in [0,1]
    let dRows = MLX.abs(img[0..., 1..., 0..., 0...] - img[0..., ..<767, 0..., 0...])
    let meanRowGrad = MLX.mean(dRows).item(Float.self)
    func boundaryRowGrad(_ y: Int) -> Float {
      let d = MLX.abs(img[0..., y..<(y + 1), 0..., 0...] - img[0..., (y - 1)..<y, 0..., 0...])
      return MLX.mean(d).item(Float.self)
    }
    let seam = Swift.max(boundaryRowGrad(256), boundaryRowGrad(512))
    let ratio = seam / Swift.max(meanRowGrad, 1e-8)
    print("[#45] seam/interior gradient ratio = \(ratio) (seam=\(seam), interior=\(meanRowGrad))")
    #expect(ratio < 3.0, "tile boundary shows a seam (gradient ratio \(ratio))")
  }

  /// The default decode dtype must match fp32 on real weights and be finite.
  ///
  /// The SDXL VAE is fp16-unstable (`force_upcast=true` in its config): an fp16
  /// decode of the real checkpoint overflows to NaN. The decoder therefore
  /// defaults to **bfloat16** — the same 16-bit footprint as fp16 (so the #45
  /// activation-memory win is preserved) but with fp32-class dynamic range, so it
  /// does not overflow. This validates that default end-to-end on real weights.
  @Test("default (bf16) decode matches fp32 and is NaN-free on real weights")
  func defaultDTypeMatchesFp32OnRealWeights() throws {
    guard let url = Self.weightsURL() else {
      Issue.record("SDXL VAE weights not found; skipping real-weights dtype parity test")
      return
    }
    // Default config (decodeDType defaults to bf16).
    let dDefault = try Self.loadDecoder(url, config: .init())
    let d32 = try Self.loadDecoder(url, config: .init(decodeDType: .float32))
    #expect(
      SDXLVAEDecoderConfiguration().decodeDType == .bfloat16, "default decode dtype must be bf16")

    MLXRandom.seed(11)
    let latent = MLXRandom.normal([1, 64, 64, 4])  // 512×512 px
    eval(latent)

    let outDefault = try dDefault.decode(latent).data
    let out32 = try d32.decode(latent).data
    eval(outDefault, out32)

    #expect(!MLX.any(MLX.isNaN(outDefault)).item(Bool.self), "default decode produced NaNs")
    #expect(!MLX.any(MLX.isInf(outDefault)).item(Bool.self), "default decode produced Infs")

    let psnr = Self.psnrDB(out32, outDefault)
    print("[#45] default(bf16)-vs-fp32 PSNR = \(psnr) dB")
    #expect(psnr > 40.0, "default decode diverges from fp32 (PSNR \(psnr) dB)")
  }

  /// Regression guard documenting *why* the default is bf16, not fp16: an fp16
  /// decode of the real (force_upcast) SDXL VAE overflows to NaN. If a future
  /// fp16-stable checkpoint is adopted this will start failing — at which point
  /// fp16 could be reconsidered as the default for the extra mantissa bits.
  @Test("fp16 decode overflows to NaN on the real force_upcast VAE (why default is bf16)")
  func fp16OverflowsOnRealWeights() throws {
    guard let url = Self.weightsURL() else {
      Issue.record("SDXL VAE weights not found; skipping fp16-instability guard")
      return
    }
    let d16 = try Self.loadDecoder(url, config: .init(decodeDType: .float16))
    MLXRandom.seed(11)
    let latent = MLXRandom.normal([1, 64, 64, 4])
    eval(latent)
    let out16 = try d16.decode(latent).data
    eval(out16)
    #expect(
      MLX.any(MLX.isNaN(out16)).item(Bool.self),
      "fp16 decode unexpectedly stable — the checkpoint may now be fp16-safe; revisit the default")
  }
}
