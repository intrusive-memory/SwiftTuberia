import CoreGraphics
import Foundation
@preconcurrency import MLX
import MLXRandom

/// The standard pipeline for all diffusion-based generation (images, video, non-speech audio).
///
/// Composed from five pipe segments: TextEncoder, Scheduler, Backbone, Decoder, Renderer.
/// The pipeline manages the lifecycle of all components and orchestrates the full
/// diffusion generation flow including CFG, img2img, and LoRA support.
///
/// Construction validates shape contracts at assembly time -- mismatched components
/// produce clear errors before any generation begins.
public actor DiffusionPipeline<
  E: TextEncoder,
  S: Scheduler,
  B: Backbone,
  D: Decoder,
  R: Renderer
>: GenerationPipeline {
  public typealias Request = DiffusionGenerationRequest
  public typealias Result = DiffusionGenerationResult

  // MARK: - Components

  let encoder: E
  let scheduler: S
  let backbone: B
  let decoder: D
  let renderer: R

  // MARK: - Recipe Metadata

  private let _supportsImageToImage: Bool
  private let _unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy
  /// Role-keyed component ID map, populated from `recipe.componentIdFor` at init time (REQ-PIPE-03).
  /// Eliminates the positional array indexing that was used in the original implementation.
  private let _componentIdByRole: [PipelineRole: String]
  private let _encoderQuantization: QuantizationConfig
  private let _backboneQuantization: QuantizationConfig
  private let _decoderQuantization: QuantizationConfig

  // MARK: - Memory Requirement (computed once from static config)

  private let _memoryRequirement: MemoryRequirement

  // MARK: - Component Readiness Seam

  /// Seam for ensuring a component is downloaded before weight loading.
  ///
  /// Defaults to the production `AcervoComponentReadinessService`. Tests may replace
  /// this with a spy by calling `pipeline.setComponentReadinessService(_:)` after
  /// construction and before `loadModels(progress:)`.
  var componentReadinessService: any ComponentReadinessService = AcervoComponentReadinessService()

  // MARK: - Memory Gate Seam (REQ-PIPE-02)

  /// Closure invoked at the start of `loadModels(progress:)` to validate available memory.
  ///
  /// Defaults to `MemoryManager.shared.hardValidate(requiredBytes:)`.
  /// Tests may replace this by calling `setMemoryGate(_:)` to inject a stub
  /// that simulates insufficient memory without touching the hardware query.
  ///
  /// Strategy: single up-front `hardValidate(peakMemoryBytes)` (REQ-PIPE-02, S4).
  /// Phased-loading with `softCheck` per phase is deferred to a future sortie —
  /// real peak-vs-phase divergence has not been observed in production workloads.
  var memoryGate: @Sendable (UInt64) async throws -> Void = { requiredBytes in
    try await MemoryManager.shared.hardValidate(requiredBytes: requiredBytes)
  }

  // MARK: - Telemetry Seam

  /// Optional telemetry reporter. Defaults to `nil` (no events emitted).
  /// Set via `setTelemetry(_:)` before calling `loadModels(progress:)` to observe load events.
  var telemetry: (any TuberiaTelemetryReporter)? = nil

  // MARK: - Init

  /// Construct a pipeline from a recipe. Calls `recipe.validate()` during construction.
  /// Throws `PipelineError.incompatibleComponents` if validation fails.
  public init<Recipe: PipelineRecipe>(recipe: Recipe) throws
  where
    Recipe.Encoder == E, Recipe.Sched == S,
    Recipe.Back == B, Recipe.Dec == D, Recipe.Rend == R
  {
    // Instantiate components from recipe configurations
    self.encoder = try E(configuration: recipe.encoderConfig)
    self.scheduler = S(configuration: recipe.schedulerConfig)
    self.backbone = try B(configuration: recipe.backboneConfig)
    self.decoder = try D(configuration: recipe.decoderConfig)
    self.renderer = R(configuration: recipe.rendererConfig)

    // Store recipe metadata
    self._supportsImageToImage = recipe.supportsImageToImage
    self._unconditionalEmbeddingStrategy = recipe.unconditionalEmbeddingStrategy
    self._componentIdByRole = recipe.componentIdFor
    self._encoderQuantization = recipe.quantizationFor(.encoder)
    self._backboneQuantization = recipe.quantizationFor(.backbone)
    self._decoderQuantization = recipe.quantizationFor(.decoder)

    // Compute memory requirement from component estimates
    let encoderMem = UInt64(encoder.estimatedMemoryBytes)
    let backboneMem = UInt64(backbone.estimatedMemoryBytes)
    let decoderMem = UInt64(decoder.estimatedMemoryBytes)
    let peakMemory = encoderMem + backboneMem + decoderMem
    // Phase 1 = encoder, Phase 2 = backbone + decoder
    let phase1Memory = encoderMem
    let phase2Memory = backboneMem + decoderMem
    let phasedMemory = max(phase1Memory, phase2Memory)
    self._memoryRequirement = MemoryRequirement(
      peakMemoryBytes: peakMemory,
      phasedMemoryBytes: phasedMemory
    )

    // Validate shape contracts at assembly time
    try Self.validateAssembly(
      encoder: encoder,
      backbone: backbone,
      decoder: decoder,
      supportsImageToImage: recipe.supportsImageToImage
    )

    // Run recipe's own validation
    try recipe.validate()
  }

  // MARK: - Assembly-Time Validation

  /// Performs six assembly-time shape contract checks.
  private static func validateAssembly(
    encoder: E,
    backbone: B,
    decoder: D,
    supportsImageToImage: Bool
  ) throws {
    // Check 1: Completeness -- components are non-nil by construction (no optionals)

    // Check 2: Encoder -> Backbone (embedding dimension)
    if encoder.outputEmbeddingDim != backbone.expectedConditioningDim {
      throw PipelineError.incompatibleComponents(
        inlet: "Backbone.expectedConditioningDim(\(backbone.expectedConditioningDim))",
        outlet: "TextEncoder.outputEmbeddingDim(\(encoder.outputEmbeddingDim))",
        reason:
          "Embedding dimension mismatch: encoder produces \(encoder.outputEmbeddingDim) but backbone expects \(backbone.expectedConditioningDim)"
      )
    }

    // Check 3: Encoder -> Backbone (sequence length)
    if encoder.maxSequenceLength != backbone.expectedMaxSequenceLength {
      throw PipelineError.incompatibleComponents(
        inlet: "Backbone.expectedMaxSequenceLength(\(backbone.expectedMaxSequenceLength))",
        outlet: "TextEncoder.maxSequenceLength(\(encoder.maxSequenceLength))",
        reason:
          "Sequence length mismatch: encoder produces \(encoder.maxSequenceLength) but backbone expects \(backbone.expectedMaxSequenceLength)"
      )
    }

    // Check 4: Backbone -> Decoder (latent channels)
    if backbone.outputLatentChannels != decoder.expectedInputChannels {
      throw PipelineError.incompatibleComponents(
        inlet: "Decoder.expectedInputChannels(\(decoder.expectedInputChannels))",
        outlet: "Backbone.outputLatentChannels(\(backbone.outputLatentChannels))",
        reason:
          "Latent channel mismatch: backbone produces \(backbone.outputLatentChannels) channels but decoder expects \(decoder.expectedInputChannels)"
      )
    }

    // Check 5: Decoder -> Renderer modality compatibility
    // This is validated implicitly by the type system in most cases.
    // The recipe's type constraints ensure compatibility.

    // Check 6: Image-to-image requires BidirectionalDecoder
    if supportsImageToImage {
      guard decoder is any BidirectionalDecoder else {
        throw PipelineError.incompatibleComponents(
          inlet: "DiffusionPipeline",
          outlet: "Decoder",
          reason:
            "Recipe declares supportsImageToImage but decoder does not conform to BidirectionalDecoder"
        )
      }
    }
  }

  /// Replace the component readiness service — used by tests to inject a spy.
  ///
  /// Call this on the actor before invoking `loadModels(progress:)`. In production
  /// code the default `AcervoComponentReadinessService` is always used.
  public func setComponentReadinessService(_ service: any ComponentReadinessService) {
    componentReadinessService = service
  }

  /// Replace the memory gate — used by tests to inject a stub that simulates
  /// insufficient memory without querying real hardware.
  ///
  /// Call this on the actor before invoking `loadModels(progress:)`.
  public func setMemoryGate(_ gate: @escaping @Sendable (UInt64) async throws -> Void) {
    memoryGate = gate
  }

  /// Replace the telemetry reporter. Default is `nil` (no events emitted).
  /// Call before `loadModels(progress:)` to capture load events.
  public func setTelemetry(_ reporter: (any TuberiaTelemetryReporter)?) {
    self.telemetry = reporter
  }

  // MARK: - GenerationPipeline Conformance

  /// Memory requirements -- computed from static config, safe to access without `await`.
  nonisolated public var memoryRequirement: MemoryRequirement {
    _memoryRequirement
  }

  /// Whether all weighted segments currently have weights loaded.
  nonisolated public var isLoaded: Bool {
    encoder.isLoaded && backbone.isLoaded && decoder.isLoaded
  }

  // MARK: - Model Loading (Two-Phase)

  /// Load all model weights using two-phase loading to manage memory.
  ///
  /// Phase 1: Load TextEncoder weights, encode any needed data, then unload.
  /// Phase 2: Load Backbone + Decoder weights for generation.
  ///
  /// Progress callback receives (fraction: Double, component: String).
  public func loadModels(progress: @escaping @Sendable (Double, String) -> Void) async throws {
    let loadStartTime = Date()

    // Telemetry: loadStart — emitted before the memory gate, payload requires actor hop.
    if let telemetry {
      let peak = _memoryRequirement.peakMemoryBytes
      let phased = _memoryRequirement.phasedMemoryBytes
      let available = await MemoryManager.shared.availableMemory
      await telemetry.capture(.loadStart(
        peakMemoryBytes: peak,
        phasedMemoryBytes: phased,
        availableMemoryBytes: available
      ))
    }

    // Memory gate (REQ-PIPE-02, S4): validate available memory before committing to loading.
    //
    // Strategy: single up-front hardValidate against peakMemoryBytes.
    // Phased-loading (softCheck per phase) is deferred — see Open Questions #5 in
    // EXECUTION_PLAN.md. Any error from hardValidate is a PipelineError.insufficientMemory
    // and surfaces directly to the caller without wrapping (MemoryManager already throws it).
    let peak = _memoryRequirement.peakMemoryBytes
    do {
      try await memoryGate(peak)
      // Telemetry: memoryGate passed — query available memory for the event payload.
      if let telemetry {
        let available = await MemoryManager.shared.availableMemory
        await telemetry.capture(.memoryGate(
          requiredBytes: peak,
          availableBytes: available,
          passed: true
        ))
      }
    } catch let error as PipelineError {
      // Telemetry: memoryGate failed, then errorThrown — emit before rethrowing.
      if case .insufficientMemory(let required, let available, _) = error {
        await emit(.memoryGate(requiredBytes: required, availableBytes: available, passed: false))
      } else {
        await emit(.memoryGate(requiredBytes: peak, availableBytes: 0, passed: false))
      }
      await emit(.errorThrown(phase: "loadModels.memoryGate", error: error))
      // Already a PipelineError (e.g. .insufficientMemory from hardValidate) — rethrow as-is.
      throw error
    } catch {
      // Unexpected error from a custom gate: wrap in insufficientMemory with 0 available.
      let wrappedError = PipelineError.insufficientMemory(required: peak, available: 0, component: "pipeline")
      await emit(.memoryGate(requiredBytes: peak, availableBytes: 0, passed: false))
      await emit(.errorThrown(phase: "loadModels.memoryGate", error: wrappedError))
      throw wrappedError
    }

    // Load the tokenizer for any encoder that supports it (e.g. T5XXLEncoder).
    // This is a non-fatal async step: if tokenizer loading fails, encode() falls
    // back to placeholder tokenization (per INF-2 Option B lifecycle design).
    if let tokenizerLoadable = encoder as? any TokenizerLoadable {
      await tokenizerLoadable.loadTokenizer()
    }

    let weightedSegments:
      [(
        segment: any WeightedSegment, componentId: String?, role: PipelineRole,
        quantization: QuantizationConfig
      )] = [
        (encoder, findComponentId(for: .encoder), .encoder, _encoderQuantization),
        (backbone, findComponentId(for: .backbone), .backbone, _backboneQuantization),
        (decoder, findComponentId(for: .decoder), .decoder, _decoderQuantization),
      ]

    let totalSegments = Double(weightedSegments.count)
    var loadedCount = 0.0
    var totalEstimatedBytes: UInt64 = 0

    for (segment, componentId, role, quantization) in weightedSegments {
      let componentName = role.rawValue
      progress(loadedCount / totalSegments, componentName)

      // Telemetry: segmentLoadStart — emitted before ensureComponentReady.
      await emit(.segmentLoadStart(
        role: role,
        componentId: componentId ?? "",
        quantization: quantization.description,
        estimatedBytes: UInt64(segment.estimatedMemoryBytes)
      ))

      if let componentId = componentId {
        // Ensure the component files are present on disk (downloads if missing).
        //
        // We fold download progress into the existing (Double, String) tick stream rather
        // than adding a new PipelineProgress case. Adding a case would widen the public
        // enum and require every switch exhausted by callers to add a new arm — a
        // breaking change. Folding keeps the contract stable (S8 has nothing new to
        // document) and is sufficient: the progress bar stays live during downloads.
        let capturedCount = loadedCount
        let capturedTotal = totalSegments
        let capturedName = componentName
        let capturedRole = role
        let capturedComponentId = componentId
        // Rate-limit segmentDownloadProgress: only emit when overallProgress advances >= 0.01.
        // _DownloadProgressState is a reference-type box allowing mutation across @Sendable
        // closure invocations (declared at file scope to work inside generic actor methods).
        let progressState = _DownloadProgressState()
        try await componentReadinessService.ensureComponentReady(componentId) { [self] downloadProgress in
          // Map Acervo's per-file fraction into the segment's slot in the overall bar.
          // Each segment occupies a 1/totalSegments window; download fill fills the
          // first half of that window, leaving the second half for weight-apply below.
          let slotStart = capturedCount / capturedTotal
          let slotWidth = 1.0 / capturedTotal
          let fraction = slotStart + slotWidth * 0.5 * downloadProgress.overallProgress
          progress(fraction, capturedName)
          // Telemetry: segmentDownloadProgress — rate-limited at the call site (>= 0.01 delta).
          // The progress closure is @Sendable but synchronous; we fire-and-forget a Task
          // to cross the async/actor boundary without blocking download progress delivery.
          let overall = downloadProgress.overallProgress
          if overall - progressState.last >= 0.01 {
            progressState.last = overall
            Task { [self] in
              await emit(.segmentDownloadProgress(
                role: capturedRole,
                componentId: capturedComponentId,
                overallProgress: overall
              ))
            }
          }
        }

        let weights = try await WeightLoader.load(
          componentId: componentId,
          keyMapping: segment.keyMapping,
          tensorTransform: segment.tensorTransform,
          quantization: quantization
        )

        // Telemetry: segmentWeightsLoaded — payload requires actor hop for bytesAfter.
        if let telemetry {
          let bytesAfter = await MemoryManager.shared.loadedComponentsMemory
          await telemetry.capture(.segmentWeightsLoaded(
            role: role,
            componentId: componentId,
            safetensorsFileCount: 1,
            parameterCount: weights.parameters.count,
            bytesAfter: bytesAfter
          ))
        }

        try segment.apply(weights: weights)

        await MemoryManager.shared.registerLoaded(
          component: componentId,
          bytes: UInt64(segment.estimatedMemoryBytes)
        )

        // Telemetry: segmentApplied — emitted after registerLoaded.
        await emit(.segmentApplied(
          role: role,
          componentId: componentId,
          estimatedMemoryBytes: UInt64(segment.estimatedMemoryBytes)
        ))

        totalEstimatedBytes += UInt64(segment.estimatedMemoryBytes)
      }

      loadedCount += 1.0
      progress(loadedCount / totalSegments, componentName)
    }

    // Telemetry: loadComplete — emitted after the loop with total duration and bytes.
    await emit(.loadComplete(
      durationSeconds: Date().timeIntervalSince(loadStartTime),
      totalEstimatedBytes: totalEstimatedBytes
    ))
  }

  /// Unload all weighted segments and unregister from MemoryManager.
  public func unloadModels() async {
    // Telemetry: unloadSegment — one per segment, emitted before the actual unload call.
    // Metal byte fields stay nil here; the post-clearGPUCache unloadSegment(role: nil, …)
    // with Metal byte deltas is wired in Sortie 9 inside MemoryManager.clearGPUCache.
    await emit(.unloadSegment(
      role: .encoder,
      componentId: _componentIdByRole[.encoder],
      metalAllocatedBytesBefore: nil,
      metalAllocatedBytesAfter: nil
    ))
    encoder.unload()

    await emit(.unloadSegment(
      role: .backbone,
      componentId: _componentIdByRole[.backbone],
      metalAllocatedBytesBefore: nil,
      metalAllocatedBytesAfter: nil
    ))
    backbone.unload()

    await emit(.unloadSegment(
      role: .decoder,
      componentId: _componentIdByRole[.decoder],
      metalAllocatedBytesBefore: nil,
      metalAllocatedBytesAfter: nil
    ))
    decoder.unload()

    for componentId in _componentIdByRole.values {
      await MemoryManager.shared.unregisterLoaded(component: componentId)
    }

    await MemoryManager.shared.clearGPUCache()
  }

  // MARK: - Generate Orchestration

  /// Execute the full diffusion generation flow.
  ///
  /// Flow:
  /// 1. Encode prompt (and negative prompt for CFG)
  /// 2. Prepare initial latents (pure noise or noised reference for img2img)
  /// 3. Configure scheduler timesteps
  /// 4. Iterative denoising loop with optional CFG
  /// 5. Decode final latents
  /// 6. Render output
  public func generate(
    request: Request,
    progress: @Sendable (PipelineProgress) -> Void
  ) async throws -> Result {
    guard encoder.isLoaded else {
      throw PipelineError.missingComponent(role: "encoder")
    }
    guard backbone.isLoaded else {
      throw PipelineError.missingComponent(role: "backbone")
    }
    guard decoder.isLoaded else {
      throw PipelineError.missingComponent(role: "decoder")
    }

    let startTime = Date()

    // Determine the actual seed
    let actualSeed = request.seed ?? UInt32.random(in: 0...UInt32.max)

    // LoRA: load adapter weights and merge into backbone before generation.
    // Single active LoRA constraint: only one LoRA config per generation call.
    // After the denoising loop, the base weights are restored via LoRALoader.unapply.
    let loraAdapterWeights: ModuleParameters?
    if let loraConfig = request.loRA {
      loraAdapterWeights = try await LoRALoader.loadAdapterWeights(
        config: loraConfig,
        keyMapping: backbone.keyMapping
      )
      // Merge LoRA adapter weights into the backbone's base weights.
      if let adapterWeights = loraAdapterWeights,
        let baseWeights = backbone.currentWeights
      {
        let mergedWeights = LoRALoader.apply(
          adapterWeights: adapterWeights,
          to: baseWeights,
          scale: loraConfig.scale
        )
        try backbone.apply(weights: mergedWeights)
      }
    } else {
      loraAdapterWeights = nil
    }

    // Prepare the prompt (with optional LoRA activation keyword)
    var effectivePrompt = request.prompt
    if let loraConfig = request.loRA, let keyword = loraConfig.activationKeyword {
      effectivePrompt = keyword + " " + effectivePrompt
    }

    // --- Step 1: Encode prompt ---
    progress(.encoding(fraction: 0.0))
    let encoderInput = TextEncoderInput(
      text: effectivePrompt,
      maxLength: encoder.maxSequenceLength
    )

    let conditionalOutput: TextEncoderOutput
    do {
      conditionalOutput = try encoder.encode(encoderInput)
    } catch {
      throw PipelineError.encodingFailed(reason: String(describing: error))
    }
    progress(.encoding(fraction: 0.5))

    // Compute unconditional embeddings for CFG
    let unconditionalOutput: TextEncoderOutput?
    let useCFG = request.guidanceScale > 1.0

    if useCFG {
      switch _unconditionalEmbeddingStrategy {
      case .emptyPrompt:
        let uncondInput = TextEncoderInput(
          text: request.negativePrompt ?? "",
          maxLength: encoder.maxSequenceLength
        )
        do {
          unconditionalOutput = try encoder.encode(uncondInput)
        } catch {
          throw PipelineError.encodingFailed(reason: "Unconditional encoding failed: \(error)")
        }

      case .zeroVector(let shape):
        let zeros = MLXArray.zeros(shape)
        let maskShape = [shape[0], shape[1]]
        let zeroMask = MLXArray.zeros(maskShape)
        unconditionalOutput = TextEncoderOutput(
          embeddings: zeros,
          mask: zeroMask
        )

      case .none:
        // No CFG -- guidance scale is embedded into the model
        unconditionalOutput = nil
      }
    } else {
      unconditionalOutput = nil
    }
    progress(.encoding(fraction: 1.0))

    // --- Step 2: Prepare initial latents ---
    let latentHeight = request.height / 8
    let latentWidth = request.width / 8
    let latentChannels = backbone.outputLatentChannels
    let latentShape = [1, latentHeight, latentWidth, latentChannels]

    // Seed the random generator
    MLXRandom.seed(UInt64(actualSeed))
    var latents: MLXArray

    // Compute img2img start timestep
    var startTimestep: Int? = nil

    if let referenceImages = request.referenceImages, !referenceImages.isEmpty,
      let strength = request.strength
    {
      // Image-to-image: encode reference, add noise
      guard let bidirectionalDecoder = decoder as? any BidirectionalDecoder else {
        throw PipelineError.generationFailed(
          step: 0,
          reason: "Image-to-image requires a BidirectionalDecoder"
        )
      }

      // Convert CGImage to MLXArray with shape [1, H, W, 3] in [0, 1] range
      let referencePixels = cgImageToMLXArray(
        referenceImages[0],
        height: request.height,
        width: request.width
      )

      do {
        let imageLatents = try bidirectionalDecoder.encode(referencePixels)
        let noise = MLXRandom.normal(latentShape)
        startTimestep = Int(Float(request.steps) * (1.0 - strength))
        let plan = scheduler.configure(steps: request.steps, startTimestep: startTimestep)
        if let firstTimestep = plan.timesteps.first {
          latents = scheduler.addNoise(to: imageLatents, noise: noise, at: firstTimestep)
        } else {
          latents = imageLatents
        }
      } catch {
        throw PipelineError.generationFailed(
          step: 0,
          reason: "Failed to encode reference image: \(error)"
        )
      }
    } else {
      // Text-to-image: pure noise
      latents = MLXRandom.normal(latentShape)
    }

    // --- Step 3: Configure scheduler ---
    scheduler.reset()
    let plan = scheduler.configure(steps: request.steps, startTimestep: startTimestep)
    let timesteps = plan.timesteps

    // --- Step 4: Denoising loop ---
    let totalSteps = timesteps.count
    for (stepIndex, timestep) in timesteps.enumerated() {
      let elapsed = Date().timeIntervalSince(startTime)
      progress(.generating(step: stepIndex + 1, totalSteps: totalSteps, elapsed: elapsed))

      let timestepArray = MLXArray(Int32(timestep))

      do {
        if useCFG, let uncondEmb = unconditionalOutput {
          // Classifier-free guidance: run backbone twice (unconditional and conditional)
          let uncondInput = BackboneInput(
            latents: latents,
            conditioning: uncondEmb.embeddings,
            conditioningMask: uncondEmb.mask,
            timestep: timestepArray
          )
          let condInput = BackboneInput(
            latents: latents,
            conditioning: conditionalOutput.embeddings,
            conditioningMask: conditionalOutput.mask,
            timestep: timestepArray
          )

          let uncondPrediction = try backbone.forward(uncondInput)
          let condPrediction = try backbone.forward(condInput)

          // CFG formula: uncond + scale * (cond - uncond)
          // Cast to float32 before scheduler math: backbone weights are float16, and at
          // high-noise timesteps (t≈999, sigma≈157) the DPM-Solver divides by sqrt(alpha_t)≈0.006,
          // amplifying float16 rounding errors 157×. Float32 prevents channel-specific bias
          // accumulation over the 20-step trajectory.
          let guidedPrediction =
            (uncondPrediction + request.guidanceScale * (condPrediction - uncondPrediction)).asType(
              .float32)

          latents = try scheduler.step(
            output: guidedPrediction,
            timestep: timestep,
            sample: latents
          )
        } else {
          // No CFG: single backbone pass
          let input = BackboneInput(
            latents: latents,
            conditioning: conditionalOutput.embeddings,
            conditioningMask: conditionalOutput.mask,
            timestep: timestepArray
          )

          // Cast to float32: see CFG branch comment above.
          let prediction = try backbone.forward(input).asType(.float32)

          latents = try scheduler.step(
            output: prediction,
            timestep: timestep,
            sample: latents
          )
        }
      } catch {
        throw PipelineError.generationFailed(
          step: stepIndex + 1,
          reason: String(describing: error)
        )
      }

      // Evaluate to ensure computation runs
      eval(latents)
    }

    // --- Step 5: Decode latents ---
    progress(.decoding)
    let decodedOutput: DecodedOutput
    do {
      decodedOutput = try decoder.decode(latents)
    } catch {
      throw PipelineError.decodingFailed(reason: String(describing: error))
    }

    // --- Step 6: Render output ---
    progress(.rendering)
    let renderedOutput: RenderedOutput
    do {
      renderedOutput = try renderer.render(decodedOutput)
    } catch {
      throw PipelineError.renderingFailed(reason: String(describing: error))
    }

    // LoRA: restore base weights after generation by subtracting the adapter delta.
    if let adapterWeights = loraAdapterWeights, let loraConfig = request.loRA,
      let currentWeights = backbone.currentWeights
    {
      let restoredWeights = LoRALoader.unapply(
        adapterWeights: adapterWeights,
        from: currentWeights,
        scale: loraConfig.scale
      )
      try backbone.apply(weights: restoredWeights)
    }

    let duration = Date().timeIntervalSince(startTime)
    progress(.complete(duration: duration))

    return DiffusionGenerationResult(
      output: renderedOutput,
      seed: actualSeed,
      steps: totalSteps,
      guidanceScale: request.guidanceScale,
      duration: duration
    )
  }

  // MARK: - Telemetry Helpers

  /// Emit a telemetry event if a reporter is set. No-op when `telemetry == nil`.
  private func emit(_ event: TuberiaTelemetryEvent) async {
    guard let telemetry else { return }
    await telemetry.capture(event)
  }

  // MARK: - Private Helpers

  /// Find the component ID for a given pipeline role using the role-keyed dictionary.
  ///
  /// Dictionary lookup eliminates positional index coupling (REQ-PIPE-03). Any role
  /// absent from the map returns `nil`, which skips weight loading for that segment.
  private func findComponentId(for role: PipelineRole) -> String? {
    _componentIdByRole[role]
  }

  /// Convert a CGImage to an MLXArray with shape [1, height, width, 3] in [0, 1] range.
  ///
  /// This function:
  /// 1. Creates a CGContext with RGB color space, 8-bit per channel
  /// 2. Draws the CGImage scaled to (width, height)
  /// 3. Extracts raw pixel bytes (handles both RGBA and RGB)
  /// 4. Converts UInt8 → Float32 and normalizes to [0, 1]
  /// 5. Drops alpha channel if present (RGB only)
  /// 6. Reshapes to [1, height, width, 3]
  ///
  /// - Parameters:
  ///   - image: The CGImage to convert
  ///   - height: Target height for resizing
  ///   - width: Target width for resizing
  /// - Returns: MLXArray with shape [1, height, width, 3] and values in [0, 1]
  private func cgImageToMLXArray(_ image: CGImage, height: Int, width: Int) -> MLXArray {
    // Create a CGContext with RGB color space, 8-bit per channel (RGBA format)
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
      // Fallback: return zeros if color space creation fails
      return MLXArray.zeros([1, height, width, 3])
    }

    let bytesPerPixel = 4  // RGBA
    let bytesPerRow = width * bytesPerPixel
    var pixelData = [UInt8](repeating: 0, count: height * width * bytesPerPixel)

    guard
      let context = CGContext(
        data: &pixelData,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      )
    else {
      // Fallback: return zeros if context creation fails
      return MLXArray.zeros([1, height, width, 3])
    }

    // Draw the CGImage scaled to the target dimensions
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    // Convert pixel data to Float32 array
    var floatPixels = [Float32](repeating: 0, count: height * width * 3)

    for y in 0..<height {
      for x in 0..<width {
        let pixelIndex = (y * width + x) * bytesPerPixel
        // Extract RGBA values
        let r = Float32(pixelData[pixelIndex]) / 255.0
        let g = Float32(pixelData[pixelIndex + 1]) / 255.0
        let b = Float32(pixelData[pixelIndex + 2]) / 255.0
        // Store as RGB (drop alpha channel)
        let rgbIndex = (y * width + x) * 3
        floatPixels[rgbIndex] = r
        floatPixels[rgbIndex + 1] = g
        floatPixels[rgbIndex + 2] = b
      }
    }

    // Create MLXArray and reshape to [1, height, width, 3]
    let array = MLXArray(floatPixels)
    let reshaped = array.reshaped([1, height, width, 3])

    return reshaped
  }
}

// MARK: - Download Progress Rate-Limit Helper

/// Mutable reference-type box used to track the last-emitted download progress value.
///
/// Declared at file scope (outside the generic actor) because Swift does not allow a
/// class type to be nested inside a generic function. Marked `@unchecked Sendable` because
/// mutations only occur serially inside `ensureComponentReady`'s progress callback.
private final class _DownloadProgressState: @unchecked Sendable {
  var last: Double = -1.0
}
