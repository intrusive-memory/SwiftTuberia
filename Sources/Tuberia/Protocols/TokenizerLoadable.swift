// MARK: - TokenizerLoadable

/// An opt-in protocol for `TextEncoder` implementations that load a tokenizer
/// asynchronously during the pipeline load phase.
///
/// The pipeline checks for this conformance in `loadModels()` and calls
/// `loadTokenizer()` alongside weight loading. This is Option B from
/// requirements/INFERENCE.md § INF-2: explicit lifecycle control with a
/// separate async step that keeps `init` synchronous.
///
/// Implementations that do not require tokenizer loading (e.g. CLIP text
/// encoder with a bundled vocab) need not conform to this protocol.
public protocol TokenizerLoadable: AnyObject, Sendable {
  /// Load the tokenizer from the component's model directory.
  ///
  /// Failure is non-fatal: encode() must fall back to placeholder
  /// tokenization when the tokenizer is not loaded.
  func loadTokenizer() async
}
