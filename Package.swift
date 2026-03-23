// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "SwiftTuberia",
    platforms: [
        .macOS(.v26),
        .iOS(.v26)
    ],
    products: [
        .library(
            name: "Tuberia",
            targets: ["Tuberia"]
        ),
        .library(
            name: "TuberiaCatalog",
            targets: ["TuberiaCatalog"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.2"),
        .package(url: "https://github.com/intrusive-memory/SwiftAcervo.git", branch: "development")
    ],
    targets: [
        .target(
            name: "Tuberia",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "SwiftAcervo", package: "SwiftAcervo")
            ]
        ),
        .target(
            name: "TuberiaCatalog",
            dependencies: [
                "Tuberia"
            ]
        ),
        .testTarget(
            name: "TuberiaTests",
            dependencies: ["Tuberia"]
        ),
        .testTarget(
            name: "TuberiaCatalogTests",
            dependencies: ["TuberiaCatalog"]
        )
    ],
    swiftLanguageModes: [.v6]
)
