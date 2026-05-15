// swift-tools-version: 6.2

import Foundation
import PackageDescription

// In CI we always pin to released remotes. Locally, prefer a sibling checkout
// at ../<name> if present so in-flight changes can be exercised end-to-end
// without publishing a release. Falls back to the remote pin if the sibling
// directory is missing, so fresh clones still build.
//
// When this manifest is evaluated as a transitive dependency inside Xcode's
// `SourcePackages/checkouts/` or SwiftPM's `.build/checkouts/`, every other
// dependency lives as a sibling in the same directory. Treating those as
// in-development local paths produces conflicting package identities, so we
// must skip the sibling shortcut in that context.
let manifestDir = (#filePath as NSString).deletingLastPathComponent
let isSPMCheckout =
  manifestDir.contains("/SourcePackages/checkouts/")
  || manifestDir.contains("/.build/checkouts/")
let isCI = ProcessInfo.processInfo.environment["CI"] == "true"
let useLocalSiblings = !isCI && !isSPMCheckout

func sibling(_ name: String, remote: String, from version: Version) -> Package.Dependency {
  let localPath = "../\(name)"
  if useLocalSiblings && FileManager.default.fileExists(atPath: localPath) {
    return .package(path: localPath)
  }
  return .package(url: remote, .upToNextMajor(from: version))
}

/// Same sibling-priority pattern as ``sibling(_:remote:from:)`` but pins to a
/// remote branch when no local sibling exists. Use only when a temporary
/// pre-release dependency on a feature branch is required; switch back to the
/// version-pinned ``sibling(_:remote:from:)`` once the upstream tags a release.
func sibling(_ name: String, remote: String, branch: String) -> Package.Dependency {
  let localPath = "../\(name)"
  if useLocalSiblings && FileManager.default.fileExists(atPath: localPath) {
    return .package(path: localPath)
  }
  return .package(url: remote, branch: branch)
}

let package = Package(
  name: "SwiftTuberia",
  platforms: [
    .macOS(.v26),
    .iOS(.v26),
  ],
  products: [
    .library(
      name: "Tuberia",
      targets: ["Tuberia"]
    ),
    .library(
      name: "TuberiaCatalog",
      targets: ["TuberiaCatalog"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMajor(from: "0.31.3")),
    sibling(
      "SwiftAcervo",
      remote: "https://github.com/intrusive-memory/SwiftAcervo.git",
      from: "0.13.0"),
    // Pinned to 0.5.x: the 0.6.0 release (2026-05-09) ships generated
    // TokenizersFFI bindings that reference RustBuffer/RustCallStatus/
    // ForeignBytes runtime types missing from the package — CI fails with
    // "cannot find type 'RustBuffer' in scope". The 0.6.2 tag ships an
    // explicit "Temporary fix for Xcode builds" commit (37f999a) the
    // maintainer flagged as a possible Xcode bug, so 0.6.x is not yet
    // stable under xcodebuild. Wait for a 0.6.x release without these
    // Xcode compile issues before bumping.
    .package(
      url: "https://github.com/DePasqualeOrg/swift-tokenizers.git",
      .upToNextMinor(from: "0.5.0")),
  ],
  targets: [
    .target(
      name: "Tuberia",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "SwiftAcervo", package: "SwiftAcervo"),
      ]
    ),
    .target(
      name: "TuberiaCatalog",
      dependencies: [
        "Tuberia",
        .product(name: "Tokenizers", package: "swift-tokenizers"),
      ]
    ),
    .testTarget(
      name: "TuberiaTests",
      dependencies: ["Tuberia"]
    ),
    .testTarget(
      name: "TuberiaGPUTests",
      dependencies: ["Tuberia"]
    ),
    .testTarget(
      name: "TuberiaCatalogTests",
      dependencies: ["TuberiaCatalog"]
    ),
    .testTarget(
      name: "TuberiaCatalogGPUTests",
      dependencies: ["TuberiaCatalog"]
    ),
  ],
  swiftLanguageModes: [.v6]
)
