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
    .executable(
      name: "VerifyComponentManifest",
      targets: ["VerifyComponentManifest"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.2"),
    .package(url: "https://github.com/intrusive-memory/SwiftAcervo.git", from: "0.7.2"),
    .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
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
        .product(name: "Transformers", package: "swift-transformers"),
      ]
    ),
    // VerifyComponentManifest — CDN manifest cross-check tool.
    // Invoked from CI after `acervo upload` to verify the uploaded manifest.json
    // matches the sha256 and sizeBytes values registered in CatalogRegistration.swift.
    // Plugin-free; depends only on TuberiaCatalog (and transitively SwiftAcervo).
    .executableTarget(
      name: "VerifyComponentManifest",
      dependencies: ["TuberiaCatalog"],
      path: "Tools/VerifyComponentManifest"
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
