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
  ///
  /// Sortie 2 (OPERATION GLASS PIPES) widened the gate signature to take a
  /// `telemetry` parameter so the default closure can forward it to
  /// `MemoryManager.hardValidate(requiredBytes:telemetry:)`. The public
  /// `setMemoryGate(_:)` seam still accepts the legacy
  /// `(UInt64) async throws -> Void` shape — internally the actor wraps any
  /// custom gate to fit the new two-argument shape so existing test stubs
  /// compile unchanged.
  var memoryGate: @Sendable (UInt64, (any TuberiaTelemetryReporter)?) async throws -> Void = {
    requiredBytes, telemetry in
    try await MemoryManager.shared.hardValidate(
      requiredBytes: requiredBytes,
      telemetry: telemetry
    )
  }

  // MARK: - Telemetry Seam (OPERATION GLASS PIPES Sortie 2)

  /// Telemetry reporter installed via `setTelemetry(_:)`.
  ///
  /// Defaults to `nil` so the pipeline is zero-cost when telemetry is off; the
  /// emission sites added in later sorties gate every `TuberiaTensorStat.sample`
  /// behind an `if let telemetry { ... }` guard so the eight MLX reductions per
  /// stat never execute when this ivar is `nil`. The reporter type uses the
  /// existential `(any TuberiaTelemetryReporter)?` form required by Swift 6.
  ///
  /// All emission sites are inside actor-isolated methods, so reading this
  /// ivar requires no further synchronization.
  ///
  /// Declared `private` per §4.1; the `setTelemetry(_:)` public surface lives
  /// in `DiffusionPipeline+Telemetry.swift` and writes through the
  /// `installTelemetry(_:)` forwarder below (cross-file Swift extensions cannot
  /// reach `private` members directly).
  private var telemetry: (any TuberiaTelemetryReporter)? = nil

  /// Internal writer for the `telemetry` ivar. Called by the public
  /// `setTelemetry(_:)` extension in `DiffusionPipeline+Telemetry.swift`.
  ///
  /// This indirection exists only because Swift `private` does not cross file
  /// boundaries: the extension file cannot assign to `self.telemetry` directly.
  /// Keeping the ivar `private` (per §4.1) and routing the write through this
  /// `internal` helper preserves the encapsulation intent.
  func installTelemetry(_ reporter: (any TuberiaTelemetryReporter)?) {
    self.telemetry = reporter
  }

  // MARK: - Init

  /// Construct a pipeline from a recipe. Calls `recipe.validate()` during construction.
  /// Throws `PipelineError.incompatibleComponents` if validation fails.
  public init<Recipe: PipelineRecipe>(
    recipe: Recipe,
    telemetry: (any TuberiaTelemetryReporter)? = nil
  ) throws
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

    // Install telemetry reporter early so init-time events are observable.
    // (setTelemetry() is still supported for late-binding callers.)
    self.telemetry = telemetry

    // Validate shape contracts at assembly time
    try Self.validateAssembly(
      encoder: encoder,
      backbone: backbone,
      decoder: decoder,
      supportsImageToImage: recipe.supportsImageToImage,
      telemetry: telemetry
    )

    // Run recipe's own validation
    try recipe.validate()

    // Emit pipelineConfigured. If a telemetry reporter was passed to init,
    // the event fires; otherwise this is a no-op (and setTelemetry() can be
    // used to bind a reporter for events emitted by generate()/loadModels()).
    if let t = telemetry {
      let recipeName = String(describing: type(of: recipe))
      let encoderType = String(describing: type(of: encoder))
      let schedulerType = String(describing: type(of: scheduler))
      let backboneType = String(describing: type(of: backbone))
      let decoderType = String(describing: type(of: decoder))
      let rendererType = String(describing: type(of: renderer))
      let encQ = "\(_encoderQuantization)"
      let bkQ = "\(_backboneQuantization)"
      let decQ = "\(_decoderQuantization)"
      let peak = _memoryRequirement.peakMemoryBytes
      let phased = _memoryRequirement.phasedMemoryBytes
      Task {
        await t.capture(
          .pipelineConfigured(
            recipeName: recipeName,
            encoderType: encoderType,
            schedulerType: schedulerType,
            backboneType: backboneType,
            decoderType: decoderType,
            rendererType: rendererType,
            encoderQuantization: encQ,
            backboneQuantization: bkQ,
            decoderQuantization: decQ,
            peakMemoryBytes: peak,
            phasedMemoryBytes: phased
          ))
      }
    }
  }

  // MARK: - Assembly-Time Validation

  /// Performs six assembly-time shape contract checks.
  ///
  /// Accepts an optional telemetry reporter so init can wire
  /// `assemblyCheckPassed` / `assemblyCheckFailed` / `errorThrown` events
  /// from inside the (synchronous) `init`. Emission uses `Task { }` so the
  /// function stays synchronous (avoiding an async-init ABI break).
  /// Tests that want to assert assembly events must `await Task.yield()`
  /// after init to let the scheduled tasks run.
  private static func validateAssembly(
    encoder: E,
    backbone: B,
    decoder: D,
    supportsImageToImage: Bool,
    telemetry: (any TuberiaTelemetryReporter)? = nil
  ) throws {
    // Check 1: Completeness -- components are non-nil by construction (no optionals).
    // Emit unconditionally; the guard is a no-op when telemetry is nil.
    if let telemetry {
      Task {
        await telemetry.capture(
          .assemblyCheckPassed(
            check: .completeness,
            inlet: String(describing: type(of: encoder)),
            outlet: String(describing: type(of: backbone))
          ))
      }
    }

    // Check 2: Encoder -> Backbone (embedding dimension)
    if encoder.outputEmbeddingDim != backbone.expectedConditioningDim {
      let inlet2 = "Backbone.expectedConditioningDim(\(backbone.expectedConditioningDim))"
      let outlet2 = "TextEncoder.outputEmbeddingDim(\(encoder.outputEmbeddingDim))"
      let reason2 =
        "Embedding dimension mismatch: encoder produces \(encoder.outputEmbeddingDim) but backbone expects \(backbone.expectedConditioningDim)"
      if let telemetry {
        Task {
          await telemetry.capture(
            .assemblyCheckFailed(
              check: .encoderToBackboneDim,
              inlet: inlet2,
              outlet: outlet2,
              reason: reason2
            ))
          await telemetry.capture(
            .errorThrown(
              phase: .assembly,
              errorDescription: reason2,
              stepIndex: nil
            ))
        }
      }
      throw PipelineError.incompatibleComponents(
        inlet: inlet2,
        outlet: outlet2,
        reason: reason2
      )
    } else {
      if let telemetry {
        Task {
          await telemetry.capture(
            .assemblyCheckPassed(
              check: .encoderToBackboneDim,
              inlet: "Backbone.expectedConditioningDim(\(backbone.expectedConditioningDim))",
              outlet: "TextEncoder.outputEmbeddingDim(\(encoder.outputEmbeddingDim))"
            ))
        }
      }
    }

    // Check 3: Encoder -> Backbone (sequence length)
    if encoder.maxSequenceLength != backbone.expectedMaxSequenceLength {
      let inlet3 = "Backbone.expectedMaxSequenceLength(\(backbone.expectedMaxSequenceLength))"
      let outlet3 = "TextEncoder.maxSequenceLength(\(encoder.maxSequenceLength))"
      let reason3 =
        "Sequence length mismatch: encoder produces \(encoder.maxSequenceLength) but backbone expects \(backbone.expectedMaxSequenceLength)"
      if let telemetry {
        Task {
          await telemetry.capture(
            .assemblyCheckFailed(
              check: .encoderToBackboneSeq,
              inlet: inlet3,
              outlet: outlet3,
              reason: reason3
            ))
          await telemetry.capture(
            .errorThrown(
              phase: .assembly,
              errorDescription: reason3,
              stepIndex: nil
            ))
        }
      }
      throw PipelineError.incompatibleComponents(
        inlet: inlet3,
        outlet: outlet3,
        reason: reason3
      )
    } else {
      if let telemetry {
        Task {
          await telemetry.capture(
            .assemblyCheckPassed(
              check: .encoderToBackboneSeq,
              inlet: "Backbone.expectedMaxSequenceLength(\(backbone.expectedMaxSequenceLength))",
              outlet: "TextEncoder.maxSequenceLength(\(encoder.maxSequenceLength))"
            ))
        }
      }
    }

    // Check 4: Backbone -> Decoder (latent channels)
    if backbone.outputLatentChannels != decoder.expectedInputChannels {
      let inlet4 = "Decoder.expectedInputChannels(\(decoder.expectedInputChannels))"
      let outlet4 = "Backbone.outputLatentChannels(\(backbone.outputLatentChannels))"
      let reason4 =
        "Latent channel mismatch: backbone produces \(backbone.outputLatentChannels) channels but decoder expects \(decoder.expectedInputChannels)"
      if let telemetry {
        Task {
          await telemetry.capture(
            .assemblyCheckFailed(
              check: .backboneToDecoder,
              inlet: inlet4,
              outlet: outlet4,
              reason: reason4
            ))
          await telemetry.capture(
            .errorThrown(
              phase: .assembly,
              errorDescription: reason4,
              stepIndex: nil
            ))
        }
      }
      throw PipelineError.incompatibleComponents(
        inlet: inlet4,
        outlet: outlet4,
        reason: reason4
      )
    } else {
      if let telemetry {
        Task {
          await telemetry.capture(
            .assemblyCheckPassed(
              check: .backboneToDecoder,
              inlet: "Decoder.expectedInputChannels(\(decoder.expectedInputChannels))",
              outlet: "Backbone.outputLatentChannels(\(backbone.outputLatentChannels))"
            ))
        }
      }
    }

    // Check 5: Decoder -> Renderer modality compatibility.
    // Validated implicitly by the type system; the recipe's type constraints
    // ensure compatibility — no runtime check needed today.
    if let telemetry {
      Task {
        await telemetry.capture(
          .assemblyCheckPassed(
            check: .decoderToRenderer,
            inlet: String(describing: type(of: decoder)),
            outlet: String(describing: type(of: decoder))
          ))
      }
    }

    // Check 6: Image-to-image requires BidirectionalDecoder
    if supportsImageToImage {
      guard decoder is any BidirectionalDecoder else {
        let inlet6 = "DiffusionPipeline"
        let outlet6 = "Decoder"
        let reason6 =
          "Recipe declares supportsImageToImage but decoder does not conform to BidirectionalDecoder"
        if let telemetry {
          Task {
            await telemetry.capture(
              .assemblyCheckFailed(
                check: .imageToImageBidirectional,
                inlet: inlet6,
                outlet: outlet6,
                reason: reason6
              ))
            await telemetry.capture(
              .errorThrown(
                phase: .assembly,
                errorDescription: reason6,
                stepIndex: nil
              ))
          }
        }
        throw PipelineError.incompatibleComponents(
          inlet: inlet6,
          outlet: outlet6,
          reason: reason6
        )
      }
      if let telemetry {
        Task {
          await telemetry.capture(
            .assemblyCheckPassed(
              check: .imageToImageBidirectional,
              inlet: "DiffusionPipeline",
              outlet: "Decoder"
            ))
        }
      }
    } else {
      // Not an img2img recipe — check passes vacuously.
      if let telemetry {
        Task {
          await telemetry.capture(
            .assemblyCheckPassed(
              check: .imageToImageBidirectional,
              inlet: "DiffusionPipeline",
              outlet: "Decoder"
            ))
        }
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
  ///
  /// Accepts the legacy `(UInt64) async throws -> Void` shape for source
  /// compatibility (Sortie 2 widened the underlying gate type to also accept a
  /// telemetry reporter — see `memoryGate`). Custom gates installed through
  /// this seam never see the telemetry reporter, by design: test stubs that
  /// simulate memory pressure have no business emitting telemetry.
  public func setMemoryGate(_ gate: @escaping @Sendable (UInt64) async throws -> Void) {
    memoryGate = { requiredBytes, _ in
      try await gate(requiredBytes)
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
  public func loadModels(progress: @escaping @Sendable (Double, String) -> Void) async throws {
    // Memory gate (REQ-PIPE-02, S4): validate available memory before committing to loading.
    //
    // Strategy: single up-front hardValidate against peakMemoryBytes.
    // Phased-loading (softCheck per phase) is deferred — see Open Questions #5 in
    // EXECUTION_PLAN.md. Any error from hardValidate is a PipelineError.insufficientMemory
    // and surfaces directly to the caller without wrapping (MemoryManager already throws it).
    let peak = _memoryRequirement.peakMemoryBytes
    do {
      // Pass the installed telemetry reporter through to the gate — the
      // default gate forwards it to `MemoryManager.hardValidate`; custom
      // gates installed via `setMemoryGate(_:)` discard the reporter (test
      // stubs do not emit telemetry).
      try await memoryGate(peak, telemetry)
      if let telemetry {
        await telemetry.capture(.memoryGateChecked(requiredBytes: peak, passed: true))
      }
    } catch let error as PipelineError {
      // Already a PipelineError (e.g. .insufficientMemory from hardValidate) — rethrow as-is.
      if let telemetry {
        await telemetry.capture(
          .memoryGateChecked(requiredBytes: peak, passed: false))
        await telemetry.capture(
          .errorThrown(
            phase: .memoryGate,
            errorDescription: String(describing: error),
            stepIndex: nil
          ))
      }
      throw error
    } catch {
      // Unexpected error from a custom gate: wrap in insufficientMemory with 0 available.
      if let telemetry {
        await telemetry.capture(
          .memoryGateChecked(requiredBytes: peak, passed: false))
        await telemetry.capture(
          .errorThrown(
            phase: .memoryGate,
            errorDescription: String(describing: error),
            stepIndex: nil
          ))
      }
      throw PipelineError.insufficientMemory(required: peak, available: 0, component: "pipeline")
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

    for (segment, componentId, role, quantization) in weightedSegments {
      let componentName = role.rawValue
      progress(loadedCount / totalSegments, componentName)

      if let componentId = componentId {
        // Emit componentReadinessChecked before we check / download the component.
        if let telemetry {
          await telemetry.capture(
            .componentReadinessChecked(componentID: componentId, ready: segment.isLoaded))
        }

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
        try await componentReadinessService.ensureComponentReady(componentId) { downloadProgress in
          // Map Acervo's per-file fraction into the segment's slot in the overall bar.
          // Each segment occupies a 1/totalSegments window; download fill fills the
          // first half of that window, leaving the second half for weight-apply below.
          let slotStart = capturedCount / capturedTotal
          let slotWidth = 1.0 / capturedTotal
          let fraction = slotStart + slotWidth * 0.5 * downloadProgress.overallProgress
          progress(fraction, capturedName)
        }

        // Emit weightLoadStart before loading weights.
        if let telemetry {
          await telemetry.capture(
            .weightLoadStart(role: componentName, componentID: componentId))
        }
        let weightLoadStart = Date()

        let weights = try await WeightLoader.load(
          componentId: componentId,
          keyMapping: segment.keyMapping,
          tensorTransform: segment.tensorTransform,
          quantization: quantization,
          telemetry: telemetry
        )

        // Emit weightLoadComplete after weights are loaded.
        if let telemetry {
          let paramCount = weights.parameters.count
          let totalBytes = weights.parameters.values.reduce(UInt64(0)) { acc, arr in
            let elementBytes: UInt64
            switch arr.dtype {
            case .float16, .bfloat16, .uint16, .int16: elementBytes = 2
            case .float32, .int32, .uint32: elementBytes = 4
            case .int8, .uint8, .bool: elementBytes = 1
            case .float64, .int64, .uint64: elementBytes = 8
            default: elementBytes = 4
            }
            return acc + UInt64(arr.size) * elementBytes
          }
          let duration = Date().timeIntervalSince(weightLoadStart)
          await telemetry.capture(
            .weightLoadComplete(
              role: componentName,
              componentID: componentId,
              paramCount: paramCount,
              totalBytes: totalBytes,
              durationSeconds: duration
            ))
        }

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
      if let telemetry {
        await telemetry.capture(
          .errorThrown(
            phase: .missingComponent,
            errorDescription: "encoder not loaded",
            stepIndex: nil
          ))
      }
      throw PipelineError.missingComponent(role: "encoder")
    }
    guard backbone.isLoaded else {
      if let telemetry {
        await telemetry.capture(
          .errorThrown(
            phase: .missingComponent,
            errorDescription: "backbone not loaded",
            stepIndex: nil
          ))
      }
      throw PipelineError.missingComponent(role: "backbone")
    }
    guard decoder.isLoaded else {
      if let telemetry {
        await telemetry.capture(
          .errorThrown(
            phase: .missingComponent,
            errorDescription: "decoder not loaded",
            stepIndex: nil
          ))
      }
      throw PipelineError.missingComponent(role: "decoder")
    }

    let startTime = Date()

    // Determine the actual seed
    let actualSeed = request.seed ?? UInt32.random(in: 0...UInt32.max)

    // Generate a run ID for telemetry correlation and emit pipelineStart.
    // pipelineEnd is emitted via do/catch below — success=true on the success
    // path, success=false in the catch (which re-throws to the caller).
    let runID = UUID()
    if let telemetry {
      await telemetry.capture(
        .pipelineStart(
          runID: runID,
          prompt: request.prompt,
          steps: request.steps,
          guidanceScale: Double(request.guidanceScale),
          seed: actualSeed,
          width: request.width,
          height: request.height
        ))
    }

    // Wrap the entire generation body in a do/catch so pipelineEnd(success:)
    // fires on both the success and error paths without using async-unsafe defer.
    do {
      // LoRA: load adapter weights and merge into backbone before generation.
      // Single active LoRA constraint: only one LoRA config per generation call.
      // After the denoising loop, the base weights are restored via LoRALoader.unapply.
      let loraAdapterWeights: ModuleParameters?
      if let loraConfig = request.loRA {
        if let telemetry {
          await telemetry.capture(
            .loraLoadStart(
              componentID: loraConfig.componentId,
              localPath: loraConfig.localPath,
              scale: Double(loraConfig.scale),
              activationKeyword: loraConfig.activationKeyword
            ))
        }
        let loraLoadStartTime = Date()

        let loaded = try await LoRALoader.loadAdapterWeights(
          config: loraConfig,
          keyMapping: backbone.keyMapping,
          telemetry: telemetry
        )
        loraAdapterWeights = loaded

        if let telemetry {
          let adapterParamCount = loaded.parameters.count
          await telemetry.capture(
            .loraLoadComplete(
              adapterParamCount: adapterParamCount,
              durationSeconds: Date().timeIntervalSince(loraLoadStartTime)
            ))
        }

        // Merge LoRA adapter weights into the backbone's base weights.
        if let adapterWeights = loraAdapterWeights,
          let baseWeights = backbone.currentWeights
        {
          let mergedWeights = LoRALoader.apply(
            adapterWeights: adapterWeights,
            to: baseWeights,
            scale: loraConfig.scale,
            telemetry: telemetry
          )
          try backbone.apply(weights: mergedWeights)

          if let telemetry {
            // Count the number of layers that were actually merged.
            let targetLayerCount = mergedWeights.parameters.count
            await telemetry.capture(.loraApplied(targetLayerCount: targetLayerCount))
          }
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
      if let telemetry {
        await telemetry.capture(
          .textEncoderForwardStart(
            role: .conditional,
            promptLength: encoderInput.text.count,
            maxLength: encoderInput.maxLength
          ))
      }
      let conditionalEncodeStart = Date()
      do {
        conditionalOutput = try encoder.encode(encoderInput)
      } catch {
        if let telemetry {
          await telemetry.capture(
            .errorThrown(
              phase: .textEncoderForward,
              errorDescription: "Conditional encoding failed: \(error)",
              stepIndex: nil
            ))
        }
        throw PipelineError.encodingFailed(reason: String(describing: error))
      }
      if let telemetry {
        let embeddingStat = TuberiaTensorStat.sample(conditionalOutput.embeddings)
        let maskStat = TuberiaTensorStat.sample(conditionalOutput.mask)
        await telemetry.capture(
          .textEncoderForwardComplete(
            role: .conditional,
            embeddingStat: embeddingStat,
            maskStat: maskStat,
            durationSeconds: Date().timeIntervalSince(conditionalEncodeStart)
          ))
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
          if let telemetry {
            await telemetry.capture(
              .textEncoderForwardStart(
                role: .unconditional,
                promptLength: uncondInput.text.count,
                maxLength: uncondInput.maxLength
              ))
          }
          let unconditionalEncodeStart = Date()
          let uncondEncodeResult: TextEncoderOutput
          do {
            uncondEncodeResult = try encoder.encode(uncondInput)
          } catch {
            if let telemetry {
              await telemetry.capture(
                .errorThrown(
                  phase: .textEncoderForward,
                  errorDescription: "Unconditional encoding failed: \(error)",
                  stepIndex: nil
                ))
            }
            throw PipelineError.encodingFailed(reason: "Unconditional encoding failed: \(error)")
          }
          unconditionalOutput = uncondEncodeResult
          if let telemetry {
            let embeddingStat = TuberiaTensorStat.sample(uncondEncodeResult.embeddings)
            let maskStat = TuberiaTensorStat.sample(uncondEncodeResult.mask)
            await telemetry.capture(
              .textEncoderForwardComplete(
                role: .unconditional,
                embeddingStat: embeddingStat,
                maskStat: maskStat,
                durationSeconds: Date().timeIntervalSince(unconditionalEncodeStart)
              ))
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

      if let telemetry {
        // Defensive head/tail slices: take min(5, count) so short schedules don't crash.
        let tsCount = plan.timesteps.count
        let tsHeadEnd = min(5, tsCount)
        let tsTailStart = max(tsCount - 5, 0)
        let timestepsHead = Array(plan.timesteps[0..<tsHeadEnd])
        let timestepsTail = Array(plan.timesteps[tsTailStart...])
        let sigCount = plan.sigmas.count
        let sigHeadEnd = min(5, sigCount)
        let sigTailStart = max(sigCount - 5, 0)
        let sigmasHead = Array(plan.sigmas[0..<sigHeadEnd])
        let sigmasTail = Array(plan.sigmas[sigTailStart...])
        await telemetry.capture(
          .schedulerConfigured(
            steps: request.steps,
            startTimestep: startTimestep,
            predictionType: scheduler.predictionType,
            timestepsHead: timestepsHead,
            timestepsTail: timestepsTail,
            sigmasHead: sigmasHead,
            sigmasTail: sigmasTail
          ))
      }

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
              (uncondPrediction + request.guidanceScale * (condPrediction - uncondPrediction))
              .asType(
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

        if let telemetry {
          let restoredLayerCount = restoredWeights.parameters.count
          await telemetry.capture(.loraUnapplied(restoredLayerCount: restoredLayerCount))
        }
      }

      let duration = Date().timeIntervalSince(startTime)
      progress(.complete(duration: duration))

      if let telemetry {
        await telemetry.capture(
          .pipelineEnd(
            runID: runID, totalSteps: totalSteps, durationSeconds: duration, success: true)
        )
      }

      return DiffusionGenerationResult(
        output: renderedOutput,
        seed: actualSeed,
        steps: totalSteps,
        guidanceScale: request.guidanceScale,
        duration: duration
      )
    } catch {
      // Error path: emit pipelineEnd(success: false) before re-throwing.
      // totalSteps may be 0 if the error occurred before the denoising loop.
      if let telemetry {
        let elapsed = Date().timeIntervalSince(startTime)
        await telemetry.capture(
          .pipelineEnd(
            runID: runID,
            totalSteps: 0,
            durationSeconds: elapsed,
            success: false
          ))
        await telemetry.capture(
          .errorThrown(
            phase: .other,
            errorDescription: String(describing: error),
            stepIndex: nil
          ))
      }
      throw error
    }
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
