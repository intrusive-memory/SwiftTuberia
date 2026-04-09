@testable import Tuberia

/// Test-only extension on DiffusionPipeline that provides helpers for loading
/// synthetic weights without downloading real weight files.
extension DiffusionPipeline {

  /// Applies empty synthetic weights to all weighted components, marking them as loaded.
  ///
  /// Use this in tests to satisfy the `isLoaded` guards in `generate()` without
  /// downloading or reading any real weight files.
  func applyEmptyWeights() throws {
    let emptyWeights = ModuleParameters(parameters: [:])
    try encoder.apply(weights: emptyWeights)
    try backbone.apply(weights: emptyWeights)
    try decoder.apply(weights: emptyWeights)
  }
}
