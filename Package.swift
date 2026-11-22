// swift-tools-version: 6.2

import PackageDescription

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
    // 0.33.0 carries the evalLock lock-order fixes (#461, #474, #499) that the
    // old exact 0.31.3 pin was waiting on (0.31.4's #410 deadlocked generation).
    .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMajor(from: "0.33.0")),
    .package(url: "https://github.com/intrusive-memory/SwiftAcervo.git", .upToNextMajor(from: "0.25.0")),
    // 0.7.1 carries upstream 0.6.3's "Fixes for Xcode build with artifact
    // bundle", so the UniFFI artifactbundle links cleanly under xcodebuild
    // (the old RustBuffer/module-map blocker is resolved).
    .package(
      url: "https://github.com/DePasqualeOrg/swift-tokenizers.git",
      .upToNextMinor(from: "0.7.5")),
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
