// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "swift-depth-estimator",
    platforms: [
        .macOS("27.0"),
    ],
    products: [
        .library(name: "DepthEstimator", targets: ["DepthEstimator"]),
    ],
    targets: [
        .target(name: "DepthEstimator"),
        .testTarget(
            name: "DepthEstimatorTests",
            dependencies: ["DepthEstimator"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
