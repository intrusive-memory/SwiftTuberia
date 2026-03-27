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

  private let encoder: E
  private let scheduler: S
  private let backbone: B
  private let decoder: D
  private let renderer: R

  // MARK: - Recipe Metadata

  private let _supportsImageToImage: Bool
  private let _unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy
  private let _allComponentIds: [String]
  private let _encoderQuantization: QuantizationConfig
  private let _backboneQuantization: QuantizationConfig
  private let _decoderQuantization: QuantizationConfig

  // MARK: - Memory Requirement (computed once from static config)

  private let _memoryRequirement: MemoryRequirement

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
    self._allComponentIds = recipe.allComponentIds
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
  public func loadModels(progress: @Sendable (Double, String) -> Void) async throws {
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

    for (segment, componentId, role, quantization) in weightedSegments {
      let componentName = role.rawValue
      progress(loadedCount / totalSegments, componentName)

      if let componentId = componentId {
        let weights = try await WeightLoader.load(
          componentId: componentId,
          keyMapping: segment.keyMapping,
          tensorTransform: segment.tensorTransform,
          quantization: quantization
        )

        try segment.apply(weights: weights)

        await MemoryManager.shared.registerLoaded(
          component: componentId,
          bytes: UInt64(segment.estimatedMemoryBytes)
        )
      }

      loadedCount += 1.0
      progress(loadedCount / totalSegments, componentName)
    }
  }

  /// Unload all weighted segments and unregister from MemoryManager.
  public func unloadModels() async {
    encoder.unload()
    backbone.unload()
    decoder.unload()

    for componentId in _allComponentIds {
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
          let guidedPrediction =
            uncondPrediction + request.guidanceScale * (condPrediction - uncondPrediction)

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

          let prediction = try backbone.forward(input)

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

  // MARK: - Private Helpers

  /// Find the component ID for a given pipeline role from the stored component IDs.
  /// This is a simplified lookup -- in practice, the recipe maps roles to component IDs.
  private func findComponentId(for role: PipelineRole) -> String? {
    // Component IDs are stored as a flat array. The recipe knows the mapping.
    // For a generalized approach, we rely on the order: encoder, backbone, decoder
    // (matching the weighted segments order). If there are more IDs than segments,
    // the extras are for scheduler/renderer which have no weights.
    switch role {
    case .encoder:
      return _allComponentIds.count > 0 ? _allComponentIds[0] : nil
    case .backbone:
      return _allComponentIds.count > 1 ? _allComponentIds[1] : nil
    case .decoder:
      return _allComponentIds.count > 2 ? _allComponentIds[2] : nil
    case .scheduler, .renderer:
      return nil
    }
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

    let bytesPerPixel = 4 // RGBA
    let bytesPerRow = width * bytesPerPixel
    var pixelData = [UInt8](repeating: 0, count: height * width * bytesPerPixel)

    guard let context = CGContext(
      data: &pixelData,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: bytesPerRow,
      space: colorSpace,
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
      // Fallback: return zeros if context creation fails
      return MLXArray.zeros([1, height, width, 3])
    }

    // Draw the CGImage scaled to the target dimensions
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    // Convert pixel data to Float32 array
    var floatPixels = [Float32](repeating: 0, count: height * width * 3)

    for y in 0 ..< height {
      for x in 0 ..< width {
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
