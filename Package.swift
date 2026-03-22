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
        .package(url: "https://github.com/intrusive-memory/SwiftAcervo.git", branch: "main")
    ],
    targets: [
        .target(
            name: "Tuberia",
            dependencies: [
                .product(name: "SwiftAcervo", package: "SwiftAcervo")
            ]
        ),
        .target(
            name: "TuberiaCatalog",
            dependencies: ["Tuberia"]
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
