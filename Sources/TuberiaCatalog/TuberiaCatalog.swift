/// TuberiaCatalog - Shared component catalog for SwiftTuberia.
///
/// Contains reusable pipe segment implementations:
/// - **Encoders**: T5XXLEncoder (4096-dim embeddings)
/// - **Schedulers**: DPMSolverScheduler (DPM-Solver++), FlowMatchEulerScheduler (FLUX)
/// - **Decoders**: SDXLVAEDecoder (4-channel latents -> pixels)
/// - **Renderers**: ImageRenderer (MLXArray -> CGImage), AudioRenderer (MLXArray -> WAV)
///
/// Import `TuberiaCatalog` to access all catalog components and their configurations.
/// Importing also triggers Acervo component registration for weighted components.
@_exported import Tuberia
