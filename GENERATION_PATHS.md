# Generation Paths — Before & After SwiftTuberia

**Purpose**: Map every generation path through SwiftVoxAlta and SwiftVinetas, showing what changes and what stays the same when SwiftTuberia is introduced.

---

## Key Distinction

SwiftVinetas uses **diffusion** (iterative denoising) — it fully adopts the DiffusionPipeline compositor.
SwiftVoxAlta uses **autoregressive TTS** (token-by-token generation) — it adopts only infrastructure, not the pipeline.

```
                    ┌─────────────────────────────┐
                    │       SwiftTuberia       │
                    │                              │
                    │  ┌────────────────────────┐  │
                    │  │   DiffusionPipeline     │  │◄── SwiftVinetas uses this
                    │  │   (compositor)          │  │
                    │  └────────────────────────┘  │
                    │                              │
                    │  ┌────────────────────────┐  │
                    │  │   Infrastructure        │  │◄── Both use this
                    │  │   WeightLoader          │  │
                    │  │   MemoryManager         │  │
                    │  │   DeviceCapability       │  │
                    │  │   LoRA System           │  │
                    │  │   Progress Reporter     │  │
                    │  └────────────────────────┘  │
                    │                              │
                    │  ┌────────────────────────┐  │
                    │  │   Catalog Components    │  │◄── SwiftVinetas uses this
                    │  │   T5-XXL, SDXL VAE     │  │    (VoxAlta does not)
                    │  │   DPM-Solver, Renderers │  │
                    │  └────────────────────────┘  │
                    └─────────────────────────────┘
```

---

## 1. SwiftVoxAlta — TTS Generation

### 1a. BEFORE (Current Architecture)

```
User: diga "Hello world" -v ryan
      │
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│  diga CLI (DigaCommand)                                              │
│  ├─ Parse args: text="Hello world", voice="ryan"                    │
│  ├─ DigaModelManager: ryan is preset → select CustomVoice model     │
│  │   └─ RAM ≥16 GB? → customVoice1_7B : customVoice0_6B            │
│  └─ Hand off to DigaEngine                                           │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  VoxAltaVoiceProvider.generateWithPresetSpeaker()                    │
│                                                                       │
│  1. Load Model ──────────────────────────────────────┐               │
│     │                                                 │               │
│     ▼                                                 ▼               │
│  ┌─────────────────────────┐    ┌──────────────────────────────┐    │
│  │  VoxAltaModelManager    │    │  SwiftAcervo v1              │    │
│  │  (actor)                │    │  ~/Library/SharedModels/     │    │
│  │                         │    │  mlx-community_Qwen3-TTS-.. │    │
│  │  • queryAvailableMemory │    │                              │    │
│  │    (Mach VM stats)      │◄──►│  • isModelAvailable()       │    │
│  │  • validateMemory()     │    │  • download() if missing    │    │
│  │    (1.5x headroom)      │    │  • modelDirectory()         │    │
│  │  • Stream.synchronize() │    │                              │    │
│  │  • Memory.clearCache()  │    └──────────────────────────────┘    │
│  └─────────────────────────┘                                         │
│                                                                       │
│  2. Generate Audio                                                    │
│     │                                                                 │
│     ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  mlx-audio-swift (Qwen3TTSModel)                            │    │
│  │                                                              │    │
│  │  Tokenize text ──► Autoregressive generation (12 Hz)        │    │
│  │  ├─ temperature=0.7, topP=0.9, repetitionPenalty=1.3       │    │
│  │  ├─ Preset speaker embedding baked into model weights       │    │
│  │  └─ Output: MLXArray [num_samples] @ 24kHz                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  3. Convert to WAV                                                    │
│     │                                                                 │
│     ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  AudioConversion.mlxArrayToWAVData()                         │    │
│  │  MLXArray → float32 → clamp [-1,1] → int16 PCM             │    │
│  │  → RIFF/WAVE header (24kHz, 16-bit, mono) → Data           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  4. Output                                                            │
│     └─► Write to file (-o flag) or play through speakers             │
└──────────────────────────────────────────────────────────────────────┘
```

**Voice Clone Path** (different entry, same infrastructure):

```
User: diga "Hello world" -v my_custom_voice
      │
      ▼
  DigaEngine → VoiceLockManager.generateAudio()
      │
      ├─ Load VoiceLock from ~/.diga/voices/
      │   └─ Contains serialized VoiceClonePrompt (speaker embedding)
      │
      ├─ Load Base model (not CustomVoice)
      │   └─ base1_7B or base0_6B (supports clone prompts)
      │
      ├─ Deserialize VoiceClonePrompt
      │   └─ Check VoxAltaVoiceCache first (avoid repeat deserialization)
      │
      ├─ Qwen3TTSModel.generateWithClonePrompt()
      │   └─ Same autoregressive pipeline, but with injected speaker embedding
      │
      └─ AudioConversion → WAV Data → output
```

**Who owns what today**:

| Responsibility | Owner |
|---|---|
| CLI parsing | diga (DigaCommand, ArgumentParser) |
| Voice routing (preset vs clone) | VoxAltaVoiceProvider |
| Model selection (0.6B vs 1.7B) | DigaModelManager / VoxAltaModelManager |
| Memory validation | VoxAltaModelManager (Mach VM queries, 1.5x headroom) |
| Device detection | AppleSiliconInfo (M1-M5 detection) |
| Model download & cache | SwiftAcervo v1 (filesystem discovery) |
| Model loading (safetensors) | mlx-audio-swift (internal to Qwen3TTSModel) |
| TTS inference | mlx-audio-swift (Qwen3TTSModel) |
| Audio conversion | AudioConversion (MLXArray → WAV) |
| Voice identity | VoiceLock + VoiceLockManager + VoxAltaVoiceCache |
| GPU memory cleanup | VoxAltaModelManager (Stream.sync + Memory.clearCache) |

---

### 1b. AFTER (With SwiftTuberia Infrastructure)

```
User: diga "Hello world" -v ryan
      │
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│  diga CLI (DigaCommand) ◄── UNCHANGED                                │
│  ├─ Parse args: text="Hello world", voice="ryan"                    │
│  ├─ DigaModelManager: ryan is preset → select CustomVoice model     │
│  └─ Hand off to DigaEngine                                           │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  VoxAltaVoiceProvider ◄── UNCHANGED public API                       │
│                                                                       │
│  1. Load Model ──────────────────────────────────────┐               │
│     │                                                 │               │
│     ▼                                                 ▼               │
│  ┌─────────────────────────┐    ┌──────────────────────────────┐    │
│  │  VoxAltaModelManager    │    │  SwiftAcervo v2              │    │
│  │  (actor)                │    │  Component Registry          │    │
│  │                         │    │                              │    │
│  │  • MemoryManager ◄─────│────│──── NEW: replaces Mach VM   │    │
│  │    (from Pipeline)      │    │        queries               │    │
│  │  • DeviceCapability ◄──│────│──── NEW: replaces            │    │
│  │    (from Pipeline)      │    │        AppleSiliconInfo      │    │
│  │  • Stream.synchronize() │    │  • ComponentDescriptor       │    │
│  │  • Memory.clearCache()  │    │    (6 TTS variants           │    │
│  │                         │    │     registered at import)    │    │
│  │                         │    │  • ensureComponentReady()    │    │
│  │                         │    │  • withComponentAccess()     │    │
│  └─────────────────────────┘    └──────────────────────────────┘    │
│                                                                       │
│  2. Generate Audio ◄── UNCHANGED                                      │
│     │                                                                 │
│     ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  mlx-audio-swift (Qwen3TTSModel) ◄── UNCHANGED              │    │
│  │  Tokenize → autoregressive → MLXArray @ 24kHz               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  3. Convert to WAV ◄── UNCHANGED                                      │
│     │                                                                 │
│     ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  AudioConversion.mlxArrayToWAVData() ◄── UNCHANGED           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  4. Output ◄── UNCHANGED                                              │
└──────────────────────────────────────────────────────────────────────┘
```

**What changes in VoxAlta** (infrastructure adoption only):

| Before | After | Why |
|---|---|---|
| `VoxAltaModelManager.queryAvailableMemory()` (Mach VM) | `MemoryManager.availableMemory` (from Pipeline) | Unified memory tracking across all loaded models |
| `AppleSiliconInfo` (M1-M5 detection) | `DeviceCapability` (from Pipeline) | One source of truth for device info |
| `Acervo.isModelAvailable(modelId)` | `Acervo.isComponentReady(componentId)` | Registry-aware: knows what exists, not just what's cached |
| `Acervo.download(modelId, files: [...])` | `Acervo.ensureComponentReady(componentId)` | No hardcoded file lists in VoxAlta code |
| Hardcoded HF repo strings in `Qwen3TTSModelRepo` | `ComponentDescriptor` registered at import | Model knowledge lives in descriptors |
| `Acervo.modelDirectory(modelId)` → file paths | `withComponentAccess { handle in ... }` | No path leakage |

**What does NOT change in VoxAlta**:

- VoiceProvider protocol conformance
- Voice routing (preset vs clone)
- mlx-audio-swift inference (Qwen3TTSModel)
- AudioConversion (MLXArray → WAV)
- VoiceLock / VoiceLockManager / VoxAltaVoiceCache
- GenerationContext / GenerationSettings
- .vox file handling
- diga CLI interface and commands
- All domain-specific logic

---

## 2. SwiftVinetas — Image Generation

### 2a. BEFORE (Current Architecture)

```
User: vinetas generate "A cat in a hat" --model klein4b
      │
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│  vinetas CLI (VinetasCLI)                                            │
│  ├─ Parse args: prompt, model=klein4b, steps, guidance, etc.        │
│  └─ Call VinetasClient.shared.generate(...)                          │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  VinetasClient                                                        │
│                                                                       │
│  1. Compose prompt: "{stylePrompt}, {panelPrompt}"                   │
│  2. Resolve engine: EngineRouter.engine(for: .klein4B)               │
│     └─ Returns Flux2Engine (engineID: "flux2")                       │
│  3. Optional: load LoRA for character consistency                     │
│  4. Build GenerationRequest                                           │
│  5. Call engine.generate(request:stepProgress:)                       │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Flux2Engine (actor)                                                  │
│                                                                       │
│  1. Create & Load Pipeline                                            │
│     │                                                                 │
│     ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Flux2Pipeline (from flux-2-swift-mlx)                       │    │
│  │                                                              │    │
│  │  loadModels(progressCallback:)                               │    │
│  │  ├─ Phase 1: Download & load Qwen3 text encoder             │    │
│  │  ├─ Phase 2: Download & load FLUX DiT transformer           │    │
│  │  └─ Phase 3: Download & load FLUX VAE decoder               │    │
│  │                                                              │    │
│  │  Each component:                                             │    │
│  │  ├─ Flux2ModelDownloader.download()                          │    │
│  │  │   └─ SwiftAcervo v1 → ~/Library/SharedModels/            │    │
│  │  ├─ Load safetensors from file paths                         │    │
│  │  ├─ Apply quantization (int4 or qint8)                       │    │
│  │  └─ Memory optimization (MemoryOptimizationConfig)           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  2. Generate Image                                                    │
│     │                                                                 │
│     ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Flux2Pipeline.generateTextToImageWithResult()               │    │
│  │                                                              │    │
│  │  ┌──────────────┐    Encode prompt                          │    │
│  │  │ Qwen3 Encoder│───► embeddings [B, seq, 4096]            │    │
│  │  └──────────────┘                                           │    │
│  │         │                                                    │    │
│  │         ▼                                                    │    │
│  │  ┌──────────────┐    Sample initial noise                   │    │
│  │  │ Flow-Match   │                                           │    │
│  │  │ Euler        │    For each timestep (20-50 steps):       │    │
│  │  │ Scheduler    │───► step(noise_pred, t, latents)          │    │
│  │  └──────────────┘         ▲                                 │    │
│  │                           │                                  │    │
│  │  ┌──────────────┐        │                                  │    │
│  │  │ FLUX DiT     │────────┘ forward(latents, cond, t)       │    │
│  │  │ (double +    │           → noise prediction              │    │
│  │  │  single      │                                           │    │
│  │  │  stream)     │                                           │    │
│  │  └──────────────┘                                           │    │
│  │         │                                                    │    │
│  │         ▼                                                    │    │
│  │  ┌──────────────┐    Decode latents                         │    │
│  │  │ FLUX VAE     │───► pixel values [B, H, W, 3]            │    │
│  │  └──────────────┘                                           │    │
│  │         │                                                    │    │
│  │         ▼                                                    │    │
│  │  ┌──────────────┐    Render image                           │    │
│  │  │ CGImage      │───► CGImage (1024×1024, etc.)             │    │
│  │  │ construction │                                           │    │
│  │  └──────────────┘                                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  3. Return Flux2GenerationResult                                      │
│     └─ Wrap as GenerationResult { image, usedPrompt, seed, duration }│
└──────────────────────────────────────────────────────────────────────┘
```

**PixArt Path (STUB — not functional today)**:

```
User: vinetas generate "A cat" --model pixart-sigma
      │
      ▼
  EngineRouter → PixArtEngine
      │
      └─ throws: "PixArt-Sigma is not available in this build"
         (behind #if canImport(PixArtCore) which doesn't exist yet)
```

**Who owns what today (Vinetas/FLUX)**:

| Responsibility | Owner |
|---|---|
| CLI parsing | vinetas (VinetasCLI, ArgumentParser) |
| Prompt composition | VinetasClient |
| Engine dispatch | EngineRouter |
| Memory validation | VinetasMemory (ProcessInfo.physicalMemory) |
| Memory strategy | VinetasMemory (sequential/balanced/resident) |
| Text encoding | Flux2Pipeline → Qwen3 encoder (internal) |
| Noise scheduling | Flux2Pipeline → Flow-Match Euler (internal) |
| Diffusion backbone | Flux2Pipeline → FLUX DiT (internal) |
| VAE decoding | Flux2Pipeline → FLUX VAE (internal) |
| Image rendering | Flux2Pipeline → CGImage construction (internal) |
| Model download | Flux2ModelDownloader → SwiftAcervo v1 |
| Weight loading | Flux2Pipeline (loads safetensors directly) |
| Quantization | Flux2Pipeline (applies int4/qint8) |
| LoRA management | Flux2Pipeline + LoRAManager |
| Progress reporting | Custom callbacks, not unified |

**The problem**: Everything below "Engine dispatch" is a monolith inside `flux-2-swift-mlx`. None of it is reusable. If we want PixArt to work, we'd have to build a second monolith.

---

### 2b. AFTER (With SwiftTuberia)

```
User: vinetas generate "A cat in a hat" --model klein4b
      │
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│  vinetas CLI (VinetasCLI) ◄── UNCHANGED                              │
│  └─ Call VinetasClient.shared.generate(...)                          │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  VinetasClient ◄── UNCHANGED public API                               │
│                                                                       │
│  1. Compose prompt ◄── UNCHANGED                                      │
│  2. Resolve engine: EngineRouter.engine(for: .klein4B)               │
│     └─ Returns Flux2Engine ◄── same dispatch, new internals          │
│  3. Build GenerationRequest ◄── UNCHANGED                             │
│  4. Call engine.generate(request:stepProgress:)                       │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Flux2Engine (~50 lines) ◄── SIMPLIFIED                               │
│                                                                       │
│  1. Assemble pipeline from recipe                                     │
│     │                                                                 │
│     ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  FluxKlein4BRecipe (from flux-2-swift-mlx)                   │    │
│  │  declares:                                                    │    │
│  │    encoder  = Qwen3TextEncoder     (from flux-2-swift-mlx)   │    │
│  │    scheduler = FlowMatchEuler      (from Catalog)            │    │
│  │    backbone = FluxDiT              (from flux-2-swift-mlx)   │    │
│  │    decoder  = FluxVAEDecoder       (from flux-2-swift-mlx)   │    │
│  │    renderer = ImageRenderer        (from Catalog)            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│     │                                                                 │
│     ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  DiffusionPipeline<Qwen3, FlowMatch, FluxDiT, FluxVAE, Img>│    │
│  │  (from SwiftTuberia)                                     │    │
│  │                                                              │    │
│  │  Assembled + validated at construction time                  │    │
│  │  Shape contracts checked (4096-dim encoder ↔ backbone)       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  2. Generate Image (orchestrated by DiffusionPipeline)                │
│     │                                                                 │
│     ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  DiffusionPipeline.generate()                                │    │
│  │                                                              │    │
│  │  ┌────────────────────────────────────────────────────────┐ │    │
│  │  │ SwiftAcervo v2: ensureComponentsReady([...])           │ │    │
│  │  │ Download any missing components before starting        │ │    │
│  │  └────────────────────────────────────────────────────────┘ │    │
│  │         │                                                    │    │
│  │         ▼                                                    │    │
│  │  ┌──────────────┐   WeightLoader loads via ComponentHandle  │    │
│  │  │ Qwen3Encoder │──► embeddings [B, seq, 4096]             │    │
│  │  │(flux-2 plugin)│   MemoryManager tracks allocation        │    │
│  │  └──────────────┘                                           │    │
│  │         │  ◄── Two-phase: unload encoder, load backbone     │    │
│  │         ▼                                                    │    │
│  │  ┌──────────────┐                                           │    │
│  │  │ FlowMatch    │   For each timestep:                      │    │
│  │  │ Euler        │──► step(noise_pred, t, latents)           │    │
│  │  │ (Catalog)    │        ▲                                  │    │
│  │  └──────────────┘        │                                  │    │
│  │                          │                                  │    │
│  │  ┌──────────────┐       │                                  │    │
│  │  │ FluxDiT      │───────┘ forward(latents, cond, t)        │    │
│  │  │(flux-2 plugin)│          → noise prediction              │    │
│  │  └──────────────┘                                           │    │
│  │         │                                                    │    │
│  │         ▼                                                    │    │
│  │  ┌──────────────┐                                           │    │
│  │  │ FluxVAE      │──► decoded pixels [B, H, W, 3]           │    │
│  │  │(flux-2 plugin)│                                          │    │
│  │  └──────────────┘                                           │    │
│  │         │                                                    │    │
│  │         ▼                                                    │    │
│  │  ┌──────────────┐                                           │    │
│  │  │ ImageRenderer│──► CGImage                                │    │
│  │  │ (Catalog)    │                                           │    │
│  │  └──────────────┘                                           │    │
│  │                                                              │    │
│  │  Progress: PipelineProgress enum (unified)                   │    │
│  │  LoRA: loaded via Pipeline LoRA system                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  3. Return GenerationResult ◄── UNCHANGED type                        │
└──────────────────────────────────────────────────────────────────────┘
```

**PixArt Path (NOW FUNCTIONAL)**:

```
User: vinetas generate "A cat" --model pixart-sigma
      │
      ▼
  VinetasClient → EngineRouter → PixArtEngine (~50 lines)
      │
      ▼
  PixArtSigmaRecipe (from pixart-swift-mlx) declares:
    encoder   = T5XXLEncoder      (from Catalog)      ◄── SHARED with future models
    scheduler = DPMSolverScheduler (from Catalog)      ◄── SHARED
    backbone  = PixArtDiT          (from pixart plugin) ◄── UNIQUE (~400 lines)
    decoder   = SDXLVAEDecoder     (from Catalog)      ◄── SHARED
    renderer  = ImageRenderer      (from Catalog)      ◄── SHARED
      │
      ▼
  DiffusionPipeline<T5XXL, DPMSolver, PixArtDiT, SDXLVAE, ImageRenderer>
      │
      ▼
  Same orchestration as FLUX above (encode → denoise → decode → render)
  but with entirely different components — assembled from recipe
      │
      ▼
  CGImage (1024×1024, iPad-viable at ~2 GB int4)
```

---

## 3. Component Ownership After Migration

### What lives WHERE:

```
┌─────────────────────────────────────────────────────────────────────┐
│  SwiftTuberia (Tuberia product)                              │
│                                                                      │
│  Protocols:                                                          │
│  ├─ TextEncoder          (inlet: String → outlet: embeddings+mask)  │
│  ├─ Scheduler            (inlet: config → outlet: timestep plan)    │
│  ├─ Backbone             (inlet: latents+cond+t → outlet: noise)    │
│  ├─ Decoder              (inlet: latents → outlet: decoded data)    │
│  ├─ Renderer             (inlet: decoded → outlet: CGImage/WAV)     │
│  ├─ PipelineRecipe       (declares which segments to connect)       │
│  └─ GenerationPipeline   (generic pipeline lifecycle)               │
│                                                                      │
│  Compositor:                                                         │
│  └─ DiffusionPipeline<E,S,B,D,R>  (orchestrates the loop)          │
│                                                                      │
│  Infrastructure:                                                     │
│  ├─ WeightLoader         (safetensors via ComponentHandle)          │
│  ├─ MemoryManager        (Mach VM + per-component tracking)         │
│  ├─ DeviceCapability     (M1-M5, memory tier, platform)             │
│  ├─ LoRA system          (load/apply/scale/unload mechanics)        │
│  └─ PipelineProgress     (unified progress enum)                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  SwiftTuberia (TuberiaCatalog product)                       │
│                                                                      │
│  Shared components (tested once, used by many models):               │
│  ├─ T5XXLEncoder         (4096-dim, ~1.2 GB int4)                   │
│  ├─ CLIPEncoder          (768-dim, ~400 MB, future)                 │
│  ├─ DPMSolverScheduler   (PixArt, SD, SDXL)                        │
│  ├─ FlowMatchEulerSched  (FLUX.2)                                   │
│  ├─ DDPMScheduler        (fallback/training)                        │
│  ├─ SDXLVAEDecoder       (4ch, scale 0.13025, ~160 MB)             │
│  ├─ ImageRenderer        (MLXArray → CGImage)                       │
│  └─ AudioRenderer        (MLXArray → WAV Data)                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  flux-2-swift-mlx (model plugin, depends on Tuberia)             │
│                                                                      │
│  Model-specific (the delta):                                         │
│  ├─ FluxDiT              (Backbone: double + single stream)         │
│  ├─ Qwen3TextEncoder     (TextEncoder: FLUX-specific encoding)      │
│  ├─ MistralTextEncoder   (TextEncoder: for FLUX Dev variant)        │
│  ├─ FluxVAEDecoder       (Decoder: FLUX-specific VAE)              │
│  ├─ FluxKlein4BRecipe    (Recipe: wires components together)        │
│  ├─ FluxKlein9BRecipe    (Recipe)                                   │
│  ├─ FluxDevRecipe        (Recipe)                                   │
│  ├─ Weight key mappings                                              │
│  ├─ Acervo ComponentDescriptors                                      │
│  └─ LoRA target declarations                                        │
│                                                                      │
│  Uses from Catalog: FlowMatchEulerScheduler, ImageRenderer          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  pixart-swift-mlx (model plugin, depends on Tuberia+Catalog)     │
│                                                                      │
│  Model-specific (the delta):                                         │
│  ├─ PixArtDiT            (Backbone: 28 blocks, 1152 hidden dim)    │
│  ├─ Weight key mapping    (~200 keys)                               │
│  ├─ PixArtSigmaRecipe    (Recipe)                                   │
│  ├─ Acervo ComponentDescriptors                                      │
│  └─ LoRA target declarations                                        │
│                                                                      │
│  Uses from Catalog: T5XXLEncoder, DPMSolverScheduler,               │
│                      SDXLVAEDecoder, ImageRenderer                  │
│  ~400 lines of new code total                                        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  SwiftVinetas (consumer, depends on plugins + Tuberia)           │
│                                                                      │
│  UNCHANGED:                                                          │
│  ├─ VinetasClient        (public API, prompt composition)           │
│  ├─ EngineRouter         (dispatch by engineID)                      │
│  ├─ ModelDescriptor      (model metadata protocol)                  │
│  ├─ StyleConfig          (generation style parameters)              │
│  ├─ Character system     (LoRA-based identity)                      │
│  ├─ Understanding module (ViT, DINOv2 — not diffusion)             │
│  └─ vinetas CLI          (all subcommands)                          │
│                                                                      │
│  SIMPLIFIED:                                                         │
│  ├─ Flux2Engine          (~50 lines, wraps DiffusionPipeline)       │
│  ├─ PixArtEngine         (~50 lines, wraps DiffusionPipeline)       │
│  └─ VinetasMemory        (delegates to MemoryManager)               │
│                                                                      │
│  NEW:                                                                │
│  └─ iPad deployment      (PixArt-only, memory-gated)               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  SwiftVoxAlta (infrastructure consumer, depends on Tuberia)      │
│                                                                      │
│  UNCHANGED:                                                          │
│  ├─ VoxAltaVoiceProvider (SwiftHablare conformance)                 │
│  ├─ VoiceLockManager     (clone prompt lifecycle)                   │
│  ├─ VoxAltaVoiceCache    (actor-based clone prompt cache)           │
│  ├─ AudioConversion      (MLXArray ↔ WAV)                          │
│  ├─ GenerationContext     (phrase + metadata envelope)              │
│  ├─ GenerationSettings   (sampling parameters)                      │
│  ├─ VoiceLock            (serialized voice identity)                │
│  ├─ mlx-audio-swift      (Qwen3TTSModel — autoregressive, not pipe)│
│  └─ diga CLI             (all commands)                              │
│                                                                      │
│  CHANGED (infrastructure swap):                                      │
│  ├─ VoxAltaModelManager  (uses MemoryManager + DeviceCapability)    │
│  ├─ Model registration   (6 ComponentDescriptors in Acervo v2)      │
│  └─ Model access         (withComponentAccess, no file paths)       │
│                                                                      │
│  REMOVED:                                                            │
│  ├─ AppleSiliconInfo     (replaced by DeviceCapability)             │
│  ├─ Hardcoded HF file lists in Qwen3TTSModelRepo enum              │
│  └─ Direct Mach VM queries in VoxAltaModelManager                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  SwiftAcervo v2 (leaf dependency, zero external deps)                │
│                                                                      │
│  v1 (unchanged):                                                     │
│  ├─ isModelAvailable()                                              │
│  ├─ listModels()                                                    │
│  ├─ download(modelId, files: [...])                                 │
│  └─ withModelAccess(modelId) { url in ... }                         │
│                                                                      │
│  v2 (additive):                                                      │
│  ├─ ComponentDescriptor  (type, HF repo, files, sizes, SHA-256)    │
│  ├─ ComponentHandle      (scoped file access, no path leakage)      │
│  ├─ register(descriptor) / unregister(id)                           │
│  ├─ ensureComponentReady(id) / ensureComponentsReady([ids])        │
│  ├─ withComponentAccess(id) { handle in ... }                       │
│  ├─ isComponentReady(id) / pendingComponents()                      │
│  └─ SHA-256 integrity verification                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Side-by-Side: FLUX vs PixArt Generation

Shows how two different models use the same pipeline with different recipes:

```
                    FLUX Klein 4B                          PixArt-Sigma XL
                    ─────────────                          ────────────────

Prompt ──────────► Qwen3TextEncoder                       T5XXLEncoder ◄──────── Prompt
                   (flux-2 plugin)                        (Catalog)
                   4096-dim embeddings                    4096-dim embeddings
                        │                                      │
                        ▼                                      ▼
                   FlowMatchEuler                         DPMSolver++
                   (Catalog)                              (Catalog)
                   ~20 steps                              ~20 steps
                        │                                      │
                        ▼                                      ▼
                   FluxDiT                                PixArtDiT
                   (flux-2 plugin)                        (pixart plugin)
                   double+single stream                   28 blocks, 1152 dim
                   ~4B params                             ~600M params
                        │                                      │
                        ▼                                      ▼
                   FluxVAE                                SDXLVAEDecoder
                   (flux-2 plugin)                        (Catalog)
                   FLUX-specific decoder                  Standard SDXL decoder
                        │                                      │
                        ▼                                      ▼
                   ImageRenderer ◄─── SAME ───►           ImageRenderer
                   (Catalog)                              (Catalog)
                        │                                      │
                        ▼                                      ▼
                   CGImage                                CGImage

Memory:  ~8 GB (int4)                            Memory: ~2 GB (int4)
Min RAM: 16 GB (macOS only)                      Min RAM: 8 GB (macOS + iPad)
Speed:   ~26s/image                              Speed:  ~10s/image (estimated)
License: Proprietary/non-commercial              License: Apache 2.0
```

**What's shared**: FlowMatchEuler, DPMSolver, ImageRenderer are catalog components tested once.
**What's unique**: Each model provides only its backbone (DiT), and optionally its own encoder/decoder.

---

## 5. The Two Kinds of Consumer

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  DIFFUSION CONSUMER (SwiftVinetas)                                │
│  Uses: DiffusionPipeline + Catalog + Infrastructure + Acervo v2  │
│                                                                   │
│  Pattern:                                                         │
│    let recipe = FluxKlein4BRecipe()                               │
│    let pipeline = try DiffusionPipeline(recipe: recipe)           │
│    let image = try await pipeline.generate(prompt, config)        │
│                                                                   │
│  The pipeline handles EVERYTHING:                                 │
│    encoding, scheduling, denoising, decoding, rendering,          │
│    weight loading, memory management, progress, LoRA              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                                                                   │
│  INFRASTRUCTURE CONSUMER (SwiftVoxAlta)                           │
│  Uses: Infrastructure + Acervo v2 ONLY (no pipeline, no catalog) │
│                                                                   │
│  Pattern:                                                         │
│    let mem = MemoryManager.shared                                 │
│    let device = DeviceCapability.current                          │
│    try await Acervo.ensureComponentReady("qwen3-tts-1.7b-cv")   │
│    try await AcervoManager.shared.withComponentAccess("...") {   │
│        let model = try Qwen3TTSModel(from: $0)  // own loading  │
│    }                                                              │
│    let audio = try await model.generate(text, voice: "ryan")     │
│                                                                   │
│  VoxAlta does its own generation — autoregressive, not diffusion │
│  Pipeline provides only shared infrastructure services            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 6. Dependency Graph Summary

```
                    Consumer Layer
            ┌───────────┐  ┌────────────┐
            │SwiftVinetas│  │SwiftVoxAlta│
            └─────┬──┬──┘  └──┬──┬──────┘
                  │  │        │  │
        ┌─────────┘  │        │  └──────────────────┐
        │            │        │                      │
        ▼            ▼        ▼                      ▼
  ┌───────────┐ ┌─────────────────┐           ┌──────────┐
  │pixart-    │ │flux-2-swift-mlx │           │mlx-audio-│
  │swift-mlx  │ │(model plugin)   │           │swift     │
  │(model     │ │                 │           │(TTS      │
  │plugin)    │ │                 │           │engine)   │
  └─────┬─────┘ └────────┬───────┘           └──────────┘
        │                │                        │
        │     ┌──────────┘                        │
        │     │                                   │
        ▼     ▼                                   │
  ┌─────────────────────────┐                    │
  │   SwiftTuberia      │◄───────────────────┘
  │   (Tuberia +        │   (infrastructure only)
  │    TuberiaCatalog)  │
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │   SwiftAcervo v2        │
  │   (zero dependencies)   │
  └─────────────────────────┘
```

**Arrows mean "depends on"**. VoxAlta depends on SwiftTuberia for infrastructure but NOT for the DiffusionPipeline or Catalog. Its TTS engine (mlx-audio-swift) remains a direct dependency.

---

## 7. What We Must Validate Before Implementation

1. **Protocol fitness**: Can the TextEncoder/Scheduler/Backbone/Decoder/Renderer protocol set express everything FLUX.2 currently does internally? Specifically:
   - FLUX's double+single stream DiT — does the Backbone inlet/outlet cover this?
   - FLUX's guidance embedding (not CFG) — where does this live?
   - Image-to-image mode — how does the pipeline handle initial image encoding?

2. **Two-phase memory**: Does unloading the encoder before loading the backbone actually work for FLUX's architecture? FLUX uses guidance-distilled models where the encoder output must persist through generation.

3. **Acervo ComponentHandle**: Can mlx-audio-swift's `Qwen3TTSModel` load from a `ComponentHandle` instead of a directory path? Or does VoxAlta need a compatibility shim?

4. **Recipe validation**: What exact shape contracts need to be checked? Embedding dimension is obvious, but what about:
   - Latent channel count (encoder channels ↔ backbone channels ↔ decoder channels)
   - Sequence length compatibility
   - Scheduler compatibility with backbone's noise prediction type (epsilon vs v-prediction vs flow matching)

5. **Progress unification**: Can VoxAlta's autoregressive token-by-token progress map onto `PipelineProgress` or does it need its own reporting path?
