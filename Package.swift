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

// Swift 7's import rule, today: a file must import every module whose members
// it uses, rather than leaning on another file's imports.
for target in package.targets where target.type != .system && target.type != .binary && target.type != .plugin {
    target.swiftSettings = (target.swiftSettings ?? []) + [.enableUpcomingFeature("MemberImportVisibility")]
}
