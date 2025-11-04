// Package.swift

import PackageDescription

let package = Package(
    name: "LongCatUnlimitedVideoGeneratorForMac",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        .app(
            name: "LongCatUnlimitedVideoGeneratorForMac",
            targets: ["LongCatUnlimitedVideoGeneratorForMac"]
        )
    ],
    dependencies: [
        // Add any dependencies here
    ],
    targets: [
        .target(
            name: "LongCatUnlimitedVideoGeneratorForMac",
            dependencies: [],
            path: "src"
        ),
        .testTarget(
            name: "LongCatUnlimitedVideoGeneratorForMacTests",
            dependencies: ["LongCatUnlimitedVideoGeneratorForMac"],
            path: "Tests"
        )
    ]
)