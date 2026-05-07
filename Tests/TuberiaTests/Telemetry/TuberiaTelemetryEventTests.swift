import Foundation
import Testing
@testable import Tuberia

// MARK: - Test-local Equatable conformances
// These allow #expect(eventA == eventB) in tests.
// Same-package (swifttuberia) conformances — no @retroactive annotation required.

extension TuberiaTelemetryEvent: Equatable {
  public static func == (lhs: TuberiaTelemetryEvent, rhs: TuberiaTelemetryEvent) -> Bool {
    switch (lhs, rhs) {
    case (.loadStart(let lp, let lph, let la), .loadStart(let rp, let rph, let ra)):
      return lp == rp && lph == rph && la == ra
    case (.memoryGate(let lr, let la, let lp), .memoryGate(let rr, let ra, let rp)):
      return lr == rr && la == ra && lp == rp
    case (
      .segmentLoadStart(let lr, let lc, let lq, let le),
      .segmentLoadStart(let rr, let rc, let rq, let re)
    ):
      return lr == rr && lc == rc && lq == rq && le == re
    case (
      .segmentDownloadProgress(let lr, let lc, let lp),
      .segmentDownloadProgress(let rr, let rc, let rp)
    ):
      return lr == rr && lc == rc && lp == rp
    case (
      .segmentWeightsLoaded(let lr, let lc, let lsf, let lpc, let lb),
      .segmentWeightsLoaded(let rr, let rc, let rsf, let rpc, let rb)
    ):
      return lr == rr && lc == rc && lsf == rsf && lpc == rpc && lb == rb
    case (
      .segmentApplied(let lr, let lc, let le),
      .segmentApplied(let rr, let rc, let re)
    ):
      return lr == rr && lc == rc && le == re
    case (.loadComplete(let ld, let lt), .loadComplete(let rd, let rt)):
      return ld == rd && lt == rt
    case (
      .unloadSegment(let lr, let lc, let lb, let la),
      .unloadSegment(let rr, let rc, let rb, let ra)
    ):
      return lr == rr && lc == rc && lb == rb && la == ra
    case (
      .generateStart(let lpc, let lw, let lh, let ls, let lg, let lri, let ll, let lse),
      .generateStart(let rpc, let rw, let rh, let rs, let rg, let rri, let rl, let rse)
    ):
      return lpc == rpc && lw == rw && lh == rh && ls == rs
        && lg == rg && lri == rri && ll == rl && lse == rse
    case (.encodeStart(let lu, let lus), .encodeStart(let ru, let rus)):
      return lu == ru && lus == rus
    case (
      .encodeComplete(let ld, let lc, let lh),
      .encodeComplete(let rd, let rc, let rh)
    ):
      return ld == rd && lc == rc && lh == rh
    case (
      .denoiseStep(let li, let lt, let lts),
      .denoiseStep(let ri, let rt, let rts)
    ):
      return li == ri && lt == rt && lts == rts
    case (.decodeStart(let ll), .decodeStart(let rl)):
      return ll == rl
    case (.decodeComplete(let ld), .decodeComplete(let rd)):
      return ld == rd
    case (.renderStart, .renderStart):
      return true
    case (.renderComplete(let ld), .renderComplete(let rd)):
      return ld == rd
    case (
      .generateComplete(let ltd, let lse, let ls),
      .generateComplete(let rtd, let rse, let rs)
    ):
      return ltd == rtd && lse == rse && ls == rs
    case (
      .loraLoadStart(let la, let ls, let lk),
      .loraLoadStart(let ra, let rs, let rk)
    ):
      return la == ra && ls == rs && lk == rk
    case (
      .loraMerge(let la, let lb, let ls),
      .loraMerge(let ra, let rb, let rs)
    ):
      return la == ra && lb == rb && ls == rs
    case (.loraUnmerge(let ls), .loraUnmerge(let rs)):
      return ls == rs
    case (.errorThrown(let lp, let le), .errorThrown(let rp, let re)):
      return lp == rp && pipelineErrorEqual(le, re)
    default:
      return false
    }
  }
}

/// Equatable helper for `PipelineError` (test-local, not production).
private func pipelineErrorEqual(_ lhs: PipelineError, _ rhs: PipelineError) -> Bool {
  switch (lhs, rhs) {
  case (
    .incompatibleComponents(let li, let lo, let lr),
    .incompatibleComponents(let ri, let ro, let rr)
  ):
    return li == ri && lo == ro && lr == rr
  case (.missingComponent(let lr), .missingComponent(let rr)):
    return lr == rr
  case (.modelNotDownloaded(let lc), .modelNotDownloaded(let rc)):
    return lc == rc
  case (
    .insufficientMemory(let lr, let la, let lc),
    .insufficientMemory(let rr, let ra, let rc)
  ):
    return lr == rr && la == ra && lc == rc
  case (.weightLoadingFailed(let lc, let lr), .weightLoadingFailed(let rc, let rr)):
    return lc == rc && lr == rr
  case (.downloadFailed(let lc, let lr), .downloadFailed(let rc, let rr)):
    return lc == rc && lr == rr
  case (.encodingFailed(let lr), .encodingFailed(let rr)):
    return lr == rr
  case (.generationFailed(let ls, let lr), .generationFailed(let rs, let rr)):
    return ls == rs && lr == rr
  case (.decodingFailed(let lr), .decodingFailed(let rr)):
    return lr == rr
  case (.renderingFailed(let lr), .renderingFailed(let rr)):
    return lr == rr
  case (.cancelled, .cancelled):
    return true
  default:
    return false
  }
}

// MARK: - Test-local JSON encoding/decoding for TuberiaTelemetryEvent
//
// Swift 6 prevents adding `Codable` conformance to an enum with associated values from
// outside the declaring file. We work around this by encoding events to/from a
// `[String: Any]` dictionary via JSONSerialization, then asserting on the decoded values.
//
// This avoids any protocol conformance declaration on production types while still
// thoroughly exercising every case's payload serialization. The approach is test-only;
// `Sources/Tuberia/Telemetry/` remains Codable-free.
//
// Encoding strategy: dictionary keyed by `"case"` (discriminator) plus one key per
// associated-value label. PipelineError is flattened inline with `"error_*"` prefixes.

private enum EventEncodeError: Error { case unknownCase }

/// Encode `event` to a JSON dictionary `[String: Any]`.
// swiftlint:disable:next cyclomatic_complexity function_body_length
private func encodeToDict(_ event: TuberiaTelemetryEvent) throws -> [String: Any] {
  var d: [String: Any] = [:]
  switch event {
  case .loadStart(let peakMemoryBytes, let phasedMemoryBytes, let availableMemoryBytes):
    d["case"] = "loadStart"
    d["peakMemoryBytes"] = peakMemoryBytes
    d["phasedMemoryBytes"] = phasedMemoryBytes
    d["availableMemoryBytes"] = availableMemoryBytes
  case .memoryGate(let requiredBytes, let availableBytes, let passed):
    d["case"] = "memoryGate"
    d["requiredBytes"] = requiredBytes
    d["availableBytes"] = availableBytes
    d["passed"] = passed
  case .segmentLoadStart(let role, let componentId, let quantization, let estimatedBytes):
    d["case"] = "segmentLoadStart"
    d["role"] = role.rawValue
    d["componentId"] = componentId
    d["quantization"] = quantization
    d["estimatedBytes"] = estimatedBytes
  case .segmentDownloadProgress(let role, let componentId, let overallProgress):
    d["case"] = "segmentDownloadProgress"
    d["role"] = role.rawValue
    d["componentId"] = componentId
    d["overallProgress"] = overallProgress
  case .segmentWeightsLoaded(
    let role, let componentId, let safetensorsFileCount, let parameterCount, let bytesAfter
  ):
    d["case"] = "segmentWeightsLoaded"
    d["role"] = role.rawValue
    d["componentId"] = componentId
    d["safetensorsFileCount"] = safetensorsFileCount
    d["parameterCount"] = parameterCount
    d["bytesAfter"] = bytesAfter
  case .segmentApplied(let role, let componentId, let estimatedMemoryBytes):
    d["case"] = "segmentApplied"
    d["role"] = role.rawValue
    d["componentId"] = componentId
    d["estimatedMemoryBytes"] = estimatedMemoryBytes
  case .loadComplete(let durationSeconds, let totalEstimatedBytes):
    d["case"] = "loadComplete"
    d["durationSeconds"] = durationSeconds
    d["totalEstimatedBytes"] = totalEstimatedBytes
  case .unloadSegment(
    let role, let componentId, let metalAllocatedBytesBefore, let metalAllocatedBytesAfter
  ):
    d["case"] = "unloadSegment"
    if let r = role { d["role"] = r.rawValue }
    if let c = componentId { d["componentId"] = c }
    if let b = metalAllocatedBytesBefore { d["metalAllocatedBytesBefore"] = b }
    if let a = metalAllocatedBytesAfter { d["metalAllocatedBytesAfter"] = a }
  case .generateStart(
    let promptCharacterCount, let width, let height, let steps,
    let guidanceScale, let hasReferenceImages, let hasLoRA, let seed
  ):
    d["case"] = "generateStart"
    d["promptCharacterCount"] = promptCharacterCount
    d["width"] = width
    d["height"] = height
    d["steps"] = steps
    d["guidanceScale"] = guidanceScale
    d["hasReferenceImages"] = hasReferenceImages
    d["hasLoRA"] = hasLoRA
    d["seed"] = seed
  case .encodeStart(let useCFG, let unconditionalStrategy):
    d["case"] = "encodeStart"
    d["useCFG"] = useCFG
    d["unconditionalStrategy"] = unconditionalStrategy
  case .encodeComplete(let durationSeconds, let conditionalEmbeddingShape, let hasUnconditional):
    d["case"] = "encodeComplete"
    d["durationSeconds"] = durationSeconds
    d["conditionalEmbeddingShape"] = conditionalEmbeddingShape
    d["hasUnconditional"] = hasUnconditional
  case .denoiseStep(let stepIndex, let totalSteps, let timestep):
    d["case"] = "denoiseStep"
    d["stepIndex"] = stepIndex
    d["totalSteps"] = totalSteps
    d["timestep"] = timestep
  case .decodeStart(let latentShape):
    d["case"] = "decodeStart"
    d["latentShape"] = latentShape
  case .decodeComplete(let durationSeconds):
    d["case"] = "decodeComplete"
    d["durationSeconds"] = durationSeconds
  case .renderStart:
    d["case"] = "renderStart"
  case .renderComplete(let durationSeconds):
    d["case"] = "renderComplete"
    d["durationSeconds"] = durationSeconds
  case .generateComplete(let totalDurationSeconds, let seed, let steps):
    d["case"] = "generateComplete"
    d["totalDurationSeconds"] = totalDurationSeconds
    d["seed"] = seed
    d["steps"] = steps
  case .loraLoadStart(let adapterPath, let scale, let activationKeyword):
    d["case"] = "loraLoadStart"
    d["adapterPath"] = adapterPath
    d["scale"] = scale
    if let k = activationKeyword { d["activationKeyword"] = k }
  case .loraMerge(let adapterParameterCount, let baseParameterCount, let scale):
    d["case"] = "loraMerge"
    d["adapterParameterCount"] = adapterParameterCount
    d["baseParameterCount"] = baseParameterCount
    d["scale"] = scale
  case .loraUnmerge(let scale):
    d["case"] = "loraUnmerge"
    d["scale"] = scale
  case .errorThrown(let phase, let error):
    d["case"] = "errorThrown"
    d["phase"] = phase
    switch error {
    case .incompatibleComponents(let inlet, let outlet, let reason):
      d["error_case"] = "incompatibleComponents"
      d["error_inlet"] = inlet
      d["error_outlet"] = outlet
      d["error_reason"] = reason
    case .missingComponent(let role):
      d["error_case"] = "missingComponent"
      d["error_role"] = role
    case .modelNotDownloaded(let component):
      d["error_case"] = "modelNotDownloaded"
      d["error_component"] = component
    case .insufficientMemory(let required, let available, let component):
      d["error_case"] = "insufficientMemory"
      d["error_required"] = required
      d["error_available"] = available
      d["error_component"] = component
    case .weightLoadingFailed(let component, let reason):
      d["error_case"] = "weightLoadingFailed"
      d["error_component"] = component
      d["error_reason"] = reason
    case .downloadFailed(let component, let reason):
      d["error_case"] = "downloadFailed"
      d["error_component"] = component
      d["error_reason"] = reason
    case .encodingFailed(let reason):
      d["error_case"] = "encodingFailed"
      d["error_reason"] = reason
    case .generationFailed(let step, let reason):
      d["error_case"] = "generationFailed"
      d["error_step"] = step
      d["error_reason"] = reason
    case .decodingFailed(let reason):
      d["error_case"] = "decodingFailed"
      d["error_reason"] = reason
    case .renderingFailed(let reason):
      d["error_case"] = "renderingFailed"
      d["error_reason"] = reason
    case .cancelled:
      d["error_case"] = "cancelled"
    }
  }
  return d
}

/// Helpers to extract typed values from a JSON dictionary.
private func str(_ d: [String: Any], _ k: String) throws -> String {
  guard let v = d[k] as? String else { throw DecodingError.valueNotFound(String.self,
    .init(codingPath: [], debugDescription: "Missing String key '\(k)'")) }
  return v
}
private func uint64(_ d: [String: Any], _ k: String) throws -> UInt64 {
  // JSONSerialization decodes integers as NSNumber; cast to UInt64.
  guard let v = d[k] as? NSNumber else { throw DecodingError.valueNotFound(UInt64.self,
    .init(codingPath: [], debugDescription: "Missing NSNumber key '\(k)'")) }
  return v.uint64Value
}
private func uint32(_ d: [String: Any], _ k: String) throws -> UInt32 {
  guard let v = d[k] as? NSNumber else { throw DecodingError.valueNotFound(UInt32.self,
    .init(codingPath: [], debugDescription: "Missing NSNumber key '\(k)'")) }
  return v.uint32Value
}
private func int(_ d: [String: Any], _ k: String) throws -> Int {
  guard let v = d[k] as? NSNumber else { throw DecodingError.valueNotFound(Int.self,
    .init(codingPath: [], debugDescription: "Missing NSNumber key '\(k)'")) }
  return v.intValue
}
private func double(_ d: [String: Any], _ k: String) throws -> Double {
  guard let v = d[k] as? NSNumber else { throw DecodingError.valueNotFound(Double.self,
    .init(codingPath: [], debugDescription: "Missing NSNumber key '\(k)'")) }
  return v.doubleValue
}
private func float(_ d: [String: Any], _ k: String) throws -> Float {
  guard let v = d[k] as? NSNumber else { throw DecodingError.valueNotFound(Float.self,
    .init(codingPath: [], debugDescription: "Missing NSNumber key '\(k)'")) }
  return v.floatValue
}
private func bool(_ d: [String: Any], _ k: String) throws -> Bool {
  guard let v = d[k] as? NSNumber else { throw DecodingError.valueNotFound(Bool.self,
    .init(codingPath: [], debugDescription: "Missing NSNumber key '\(k)'")) }
  return v.boolValue
}
private func intArray(_ d: [String: Any], _ k: String) throws -> [Int] {
  guard let v = d[k] as? [NSNumber] else { throw DecodingError.valueNotFound([Int].self,
    .init(codingPath: [], debugDescription: "Missing [NSNumber] key '\(k)'")) }
  return v.map { $0.intValue }
}
private func role(_ d: [String: Any], _ k: String) throws -> PipelineRole {
  let raw = try str(d, k)
  guard let r = PipelineRole(rawValue: raw) else {
    throw DecodingError.dataCorrupted(.init(codingPath: [],
      debugDescription: "Unknown PipelineRole raw value: \(raw)"))
  }
  return r
}

/// Decode a `TuberiaTelemetryEvent` from a JSON dictionary.
// swiftlint:disable:next cyclomatic_complexity function_body_length
private func decodeFromDict(_ d: [String: Any]) throws -> TuberiaTelemetryEvent {
  let caseName = try str(d, "case")
  switch caseName {
  case "loadStart":
    return .loadStart(
      peakMemoryBytes: try uint64(d, "peakMemoryBytes"),
      phasedMemoryBytes: try uint64(d, "phasedMemoryBytes"),
      availableMemoryBytes: try uint64(d, "availableMemoryBytes")
    )
  case "memoryGate":
    return .memoryGate(
      requiredBytes: try uint64(d, "requiredBytes"),
      availableBytes: try uint64(d, "availableBytes"),
      passed: try bool(d, "passed")
    )
  case "segmentLoadStart":
    return .segmentLoadStart(
      role: try role(d, "role"),
      componentId: try str(d, "componentId"),
      quantization: try str(d, "quantization"),
      estimatedBytes: try uint64(d, "estimatedBytes")
    )
  case "segmentDownloadProgress":
    return .segmentDownloadProgress(
      role: try role(d, "role"),
      componentId: try str(d, "componentId"),
      overallProgress: try double(d, "overallProgress")
    )
  case "segmentWeightsLoaded":
    return .segmentWeightsLoaded(
      role: try role(d, "role"),
      componentId: try str(d, "componentId"),
      safetensorsFileCount: try int(d, "safetensorsFileCount"),
      parameterCount: try int(d, "parameterCount"),
      bytesAfter: try uint64(d, "bytesAfter")
    )
  case "segmentApplied":
    return .segmentApplied(
      role: try role(d, "role"),
      componentId: try str(d, "componentId"),
      estimatedMemoryBytes: try uint64(d, "estimatedMemoryBytes")
    )
  case "loadComplete":
    return .loadComplete(
      durationSeconds: try double(d, "durationSeconds"),
      totalEstimatedBytes: try uint64(d, "totalEstimatedBytes")
    )
  case "unloadSegment":
    let rawRole = d["role"] as? String
    let decodedRole = rawRole.flatMap { PipelineRole(rawValue: $0) }
    let metalBefore = (d["metalAllocatedBytesBefore"] as? NSNumber).map { $0.uint64Value }
    let metalAfter = (d["metalAllocatedBytesAfter"] as? NSNumber).map { $0.uint64Value }
    return .unloadSegment(
      role: decodedRole,
      componentId: d["componentId"] as? String,
      metalAllocatedBytesBefore: metalBefore,
      metalAllocatedBytesAfter: metalAfter
    )
  case "generateStart":
    return .generateStart(
      promptCharacterCount: try int(d, "promptCharacterCount"),
      width: try int(d, "width"),
      height: try int(d, "height"),
      steps: try int(d, "steps"),
      guidanceScale: try float(d, "guidanceScale"),
      hasReferenceImages: try bool(d, "hasReferenceImages"),
      hasLoRA: try bool(d, "hasLoRA"),
      seed: try uint32(d, "seed")
    )
  case "encodeStart":
    return .encodeStart(
      useCFG: try bool(d, "useCFG"),
      unconditionalStrategy: try str(d, "unconditionalStrategy")
    )
  case "encodeComplete":
    return .encodeComplete(
      durationSeconds: try double(d, "durationSeconds"),
      conditionalEmbeddingShape: try intArray(d, "conditionalEmbeddingShape"),
      hasUnconditional: try bool(d, "hasUnconditional")
    )
  case "denoiseStep":
    return .denoiseStep(
      stepIndex: try int(d, "stepIndex"),
      totalSteps: try int(d, "totalSteps"),
      timestep: try int(d, "timestep")
    )
  case "decodeStart":
    return .decodeStart(latentShape: try intArray(d, "latentShape"))
  case "decodeComplete":
    return .decodeComplete(durationSeconds: try double(d, "durationSeconds"))
  case "renderStart":
    return .renderStart
  case "renderComplete":
    return .renderComplete(durationSeconds: try double(d, "durationSeconds"))
  case "generateComplete":
    return .generateComplete(
      totalDurationSeconds: try double(d, "totalDurationSeconds"),
      seed: try uint32(d, "seed"),
      steps: try int(d, "steps")
    )
  case "loraLoadStart":
    return .loraLoadStart(
      adapterPath: try str(d, "adapterPath"),
      scale: try float(d, "scale"),
      activationKeyword: d["activationKeyword"] as? String
    )
  case "loraMerge":
    return .loraMerge(
      adapterParameterCount: try int(d, "adapterParameterCount"),
      baseParameterCount: try int(d, "baseParameterCount"),
      scale: try float(d, "scale")
    )
  case "loraUnmerge":
    return .loraUnmerge(scale: try float(d, "scale"))
  case "errorThrown":
    let phase = try str(d, "phase")
    let errorCase = try str(d, "error_case")
    let pipelineError: PipelineError
    switch errorCase {
    case "incompatibleComponents":
      pipelineError = .incompatibleComponents(
        inlet: try str(d, "error_inlet"),
        outlet: try str(d, "error_outlet"),
        reason: try str(d, "error_reason")
      )
    case "missingComponent":
      pipelineError = .missingComponent(role: try str(d, "error_role"))
    case "modelNotDownloaded":
      pipelineError = .modelNotDownloaded(component: try str(d, "error_component"))
    case "insufficientMemory":
      pipelineError = .insufficientMemory(
        required: try uint64(d, "error_required"),
        available: try uint64(d, "error_available"),
        component: try str(d, "error_component")
      )
    case "weightLoadingFailed":
      pipelineError = .weightLoadingFailed(
        component: try str(d, "error_component"),
        reason: try str(d, "error_reason")
      )
    case "downloadFailed":
      pipelineError = .downloadFailed(
        component: try str(d, "error_component"),
        reason: try str(d, "error_reason")
      )
    case "encodingFailed":
      pipelineError = .encodingFailed(reason: try str(d, "error_reason"))
    case "generationFailed":
      pipelineError = .generationFailed(
        step: try int(d, "error_step"),
        reason: try str(d, "error_reason")
      )
    case "decodingFailed":
      pipelineError = .decodingFailed(reason: try str(d, "error_reason"))
    case "renderingFailed":
      pipelineError = .renderingFailed(reason: try str(d, "error_reason"))
    case "cancelled":
      pipelineError = .cancelled
    default:
      throw DecodingError.dataCorrupted(.init(codingPath: [],
        debugDescription: "Unknown PipelineError case: \(errorCase)"))
    }
    return .errorThrown(phase: phase, error: pipelineError)
  default:
    throw DecodingError.dataCorrupted(.init(codingPath: [],
      debugDescription: "Unknown TuberiaTelemetryEvent case: \(caseName)"))
  }
}

// MARK: - Helper: round-trip through JSONSerialization

private func roundTrip(_ event: TuberiaTelemetryEvent) throws -> TuberiaTelemetryEvent {
  let dict = try encodeToDict(event)
  let data = try JSONSerialization.data(withJSONObject: dict)
  let decoded = try JSONSerialization.jsonObject(with: data) as? [String: Any]
  guard let decoded else {
    throw DecodingError.dataCorrupted(.init(codingPath: [],
      debugDescription: "JSONSerialization returned non-dictionary"))
  }
  return try decodeFromDict(decoded)
}

// MARK: - Test Suite

@Suite("TuberiaTelemetryEvent Tests", .serialized)
struct TuberiaTelemetryEventTests {

  // MARK: (a) Sendable transfer across actor boundary

  @Test("Sendable: event crosses actor boundary into MockTuberiaTelemetryReporter")
  func sendableActorBoundary() async throws {
    let mock = MockTuberiaTelemetryReporter()

    // Store an event in a `let` — proves it's value-typed and Sendable.
    let event: TuberiaTelemetryEvent = .loadStart(
      peakMemoryBytes: 1_000_000,
      phasedMemoryBytes: 2_000_000,
      availableMemoryBytes: 8_000_000_000
    )

    // Hand it across the async (actor hop) boundary.
    await mock.capture(event)

    let recorded = await mock.events
    #expect(recorded.count == 1)
    guard case .loadStart(
      let peak, let phased, let available
    ) = try #require(recorded.first) else {
      Issue.record("Expected .loadStart but got \(String(describing: recorded.first))")
      return
    }
    #expect(peak == 1_000_000)
    #expect(phased == 2_000_000)
    #expect(available == 8_000_000_000)
  }

  @Test("MockTuberiaTelemetryReporter records multiple events in order")
  func mockRecordsEventsInOrder() async throws {
    let mock = MockTuberiaTelemetryReporter()
    await mock.capture(.renderStart)
    await mock.capture(.decodeComplete(durationSeconds: 0.5))
    await mock.capture(.renderComplete(durationSeconds: 1.0))

    let events = await mock.events
    #expect(events.count == 3)
    #expect(events[0] == .renderStart)
    #expect(events[1] == .decodeComplete(durationSeconds: 0.5))
    #expect(events[2] == .renderComplete(durationSeconds: 1.0))
  }

  @Test("MockTuberiaTelemetryReporter drain clears buffer")
  func mockDrainClearsBuffer() async {
    let mock = MockTuberiaTelemetryReporter()
    await mock.capture(.renderStart)
    let drained = await mock.drain()
    let remaining = await mock.events
    #expect(drained.count == 1)
    #expect(remaining.isEmpty)
  }

  // MARK: (b) Codable round-trips — one per enum case

  @Test("Codable round-trip: loadStart")
  func roundTripLoadStart() throws {
    let original = TuberiaTelemetryEvent.loadStart(
      peakMemoryBytes: 4_294_967_295,
      phasedMemoryBytes: 1_073_741_824,
      availableMemoryBytes: 17_179_869_184
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: memoryGate passed")
  func roundTripMemoryGatePassed() throws {
    let original = TuberiaTelemetryEvent.memoryGate(
      requiredBytes: 2_000_000_000,
      availableBytes: 8_000_000_000,
      passed: true
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: memoryGate failed")
  func roundTripMemoryGateFailed() throws {
    let original = TuberiaTelemetryEvent.memoryGate(
      requiredBytes: 10_000_000_000,
      availableBytes: 4_000_000_000,
      passed: false
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: segmentLoadStart")
  func roundTripSegmentLoadStart() throws {
    let original = TuberiaTelemetryEvent.segmentLoadStart(
      role: .encoder,
      componentId: "encoder-q4",
      quantization: "int4",
      estimatedBytes: 512_000_000
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: segmentDownloadProgress")
  func roundTripSegmentDownloadProgress() throws {
    let original = TuberiaTelemetryEvent.segmentDownloadProgress(
      role: .backbone,
      componentId: "backbone-q8",
      overallProgress: 0.42
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: segmentWeightsLoaded")
  func roundTripSegmentWeightsLoaded() throws {
    let original = TuberiaTelemetryEvent.segmentWeightsLoaded(
      role: .decoder,
      componentId: "vae-decoder",
      safetensorsFileCount: 3,
      parameterCount: 83_500_000,
      bytesAfter: 1_073_741_824
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: segmentApplied")
  func roundTripSegmentApplied() throws {
    let original = TuberiaTelemetryEvent.segmentApplied(
      role: .scheduler,
      componentId: "scheduler-v1",
      estimatedMemoryBytes: 4_096
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: loadComplete")
  func roundTripLoadComplete() throws {
    let original = TuberiaTelemetryEvent.loadComplete(
      durationSeconds: 12.345,
      totalEstimatedBytes: 4_294_967_296
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: unloadSegment with role")
  func roundTripUnloadSegmentWithRole() throws {
    let original = TuberiaTelemetryEvent.unloadSegment(
      role: .encoder,
      componentId: "encoder-q4",
      metalAllocatedBytesBefore: 512_000_000,
      metalAllocatedBytesAfter: 0
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: unloadSegment post-cache-clear (role nil)")
  func roundTripUnloadSegmentNilRole() throws {
    let original = TuberiaTelemetryEvent.unloadSegment(
      role: nil,
      componentId: nil,
      metalAllocatedBytesBefore: 2_000_000_000,
      metalAllocatedBytesAfter: 64_000_000
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: generateStart")
  func roundTripGenerateStart() throws {
    let original = TuberiaTelemetryEvent.generateStart(
      promptCharacterCount: 42,
      width: 512,
      height: 512,
      steps: 20,
      guidanceScale: 7.5,
      hasReferenceImages: false,
      hasLoRA: true,
      seed: 12345
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: encodeStart")
  func roundTripEncodeStart() throws {
    let original = TuberiaTelemetryEvent.encodeStart(
      useCFG: true,
      unconditionalStrategy: "emptyPrompt"
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: encodeComplete")
  func roundTripEncodeComplete() throws {
    let original = TuberiaTelemetryEvent.encodeComplete(
      durationSeconds: 0.125,
      conditionalEmbeddingShape: [1, 77, 768],
      hasUnconditional: true
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: denoiseStep")
  func roundTripDenoiseStep() throws {
    let original = TuberiaTelemetryEvent.denoiseStep(
      stepIndex: 9,
      totalSteps: 20,
      timestep: 500
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: decodeStart")
  func roundTripDecodeStart() throws {
    let original = TuberiaTelemetryEvent.decodeStart(latentShape: [1, 64, 64, 4])
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: decodeComplete")
  func roundTripDecodeComplete() throws {
    let original = TuberiaTelemetryEvent.decodeComplete(durationSeconds: 0.88)
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: renderStart")
  func roundTripRenderStart() throws {
    let original = TuberiaTelemetryEvent.renderStart
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: renderComplete")
  func roundTripRenderComplete() throws {
    let original = TuberiaTelemetryEvent.renderComplete(durationSeconds: 0.012)
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: generateComplete")
  func roundTripGenerateComplete() throws {
    let original = TuberiaTelemetryEvent.generateComplete(
      totalDurationSeconds: 45.6,
      seed: 99999,
      steps: 30
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: loraLoadStart with keyword")
  func roundTripLoraLoadStartWithKeyword() throws {
    let original = TuberiaTelemetryEvent.loraLoadStart(
      adapterPath: "/models/lora/style.safetensors",
      scale: 0.8,
      activationKeyword: "ohwx"
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: loraLoadStart without keyword")
  func roundTripLoraLoadStartNoKeyword() throws {
    let original = TuberiaTelemetryEvent.loraLoadStart(
      adapterPath: "/models/lora/no-keyword.safetensors",
      scale: 1.0,
      activationKeyword: nil
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: loraMerge")
  func roundTripLoraMerge() throws {
    let original = TuberiaTelemetryEvent.loraMerge(
      adapterParameterCount: 2_000_000,
      baseParameterCount: 900_000_000,
      scale: 0.7
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: loraUnmerge")
  func roundTripLoraUnmerge() throws {
    let original = TuberiaTelemetryEvent.loraUnmerge(scale: 0.7)
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: errorThrown with insufficientMemory")
  func roundTripErrorThrown() throws {
    let original = TuberiaTelemetryEvent.errorThrown(
      phase: "loadModels.memoryGate",
      error: .insufficientMemory(
        required: 8_000_000_000,
        available: 4_000_000_000,
        component: "backbone"
      )
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }

  @Test("Codable round-trip: errorThrown cancelled")
  func roundTripErrorThrownCancelled() throws {
    let original = TuberiaTelemetryEvent.errorThrown(
      phase: "generate.denoise.step.5",
      error: .cancelled
    )
    let decoded = try roundTrip(original)
    #expect(decoded == original)
  }
}
