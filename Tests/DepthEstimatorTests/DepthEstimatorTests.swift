//
//  DepthEstimatorTests.swift
//  DepthEstimatorTests
//
//  Created by David Sherlock on 2026.
//
//  The Depth Anything V2 port against the authors' `infer_image`: the size
//  rule and the prepared tensor on fixtures the export wrote (no model
//  needed), then — with the asset installed under ~/Library/Application
//  Support/img — the network at every exported size and whole images end
//  to end, upsampling included.
//

import Foundation
import Testing
@testable import DepthEstimator

private func fixtureURL(_ name: String, _ ext: String) throws -> URL {
    try #require(Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Fixtures"))
}
private func floats(_ name: String) throws -> [Float] {
    try Data(contentsOf: fixtureURL(name, "f32")).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}
private func psnr(_ reference: [Float], _ got: [Float]) -> Double {
    precondition(reference.count == got.count)
    var err = 0.0, lo = Double.infinity, hi = -Double.infinity
    for i in reference.indices { let d = Double(reference[i]) - Double(got[i]); err += d * d; lo = min(lo, Double(reference[i])); hi = max(hi, Double(reference[i])) }
    let mse = err / Double(reference.count)
    return mse == 0 ? .infinity : 20 * log10((hi - lo) / mse.squareRoot())
}
private struct Manifest: Decodable {
    struct Image: Decodable { let width: Int; let height: Int; let input: String; let exported: Bool; let range: [Float] }
    let sizes: [String]; let images: [String: Image]
}
private func manifest() throws -> Manifest { try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: fixtureURL("manifest", "json"))) }
private func rgb(_ name: String, _ m: Manifest.Image) throws -> [UInt8] { Array(try Data(contentsOf: fixtureURL(name, "rgb"))) }
private let installed = URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support/img").appendingPathComponent(DepthEstimator.assetName)
private var modelInstalled: Bool { FileManager.default.fileExists(atPath: installed.path) }

@Suite struct InputTests {

    @Test("the size rule picks the authors' size for every fixture image and the probe's cases")
    func sizes() throws {
        let m = try manifest()
        for (name, image) in m.images { #expect(DepthInput.Size.reference(for: image.width, image.height).name == image.input, Comment(rawValue: name)) }
        let cases: [(Int, Int, String)] = [(1024, 1024, "h518w518"), (1920, 1080, "h518w924"), (1080, 1920, "h924w518"), (1024, 768, "h518w686"),
                                           (1500, 1000, "h518w784"), (800, 600, "h518w686"), (640, 400, "h518w826"), (300, 200, "h518w784")]
        for (w, h, expected) in cases { #expect(DepthInput.Size.reference(for: w, h).name == expected, "\(w)x\(h)") }
    }

    @Test("the prepared tensor matches image2tensor", arguments: ["street", "small", "square", "wide169", "tall916"])
    func tensor(name: String) throws {
        let m = try manifest()
        let image = try #require(m.images[name])
        let size = DepthEstimator.size(fromName: image.input)!
        let input = DepthInput(rgb: try rgb(name, image), width: image.width, height: image.height, size: size)
        let reference = try floats("\(name)_input")
        #expect(input.tensor.count == reference.count)
        let db = psnr(reference, input.tensor)
        let maxDiff = zip(reference, input.tensor).map { abs($0 - $1) }.max()!
        print("input \(name) (\(image.width)x\(image.height) → \(image.input)): \(String(format: "%.1f", db)) dB, max |diff| \(maxDiff)")
        #expect(db > 100)
    }

    @Test("resamplers: identity, constants, and torch's bilinear on a ramp")
    func resamplers() {
        let constant = [Double](repeating: 0.3, count: 10 * 7 * 3)
        let up = Resample.cubic(constant, width: 10, height: 7, channels: 3, toWidth: 23, toHeight: 15)
        #expect(up.allSatisfy { abs($0 - 0.3) < 1e-12 })
        let ramp: [Float] = [0, 1, 2, 3]   // 2×2
        let big = Resample.bilinearAlignCorners(ramp, width: 2, height: 2, toWidth: 3, toHeight: 3)
        #expect(big == [0, 0.5, 1, 1, 1.5, 2, 2, 2.5, 3])
        #expect(Resample.bilinearAlignCorners(ramp, width: 2, height: 2, toWidth: 2, toHeight: 2) == ramp)
    }
}

@Suite(.serialized) struct ModelTests {

    @Test("the network matches the authors' model at every fixture size", .enabled(if: modelInstalled), arguments: ["street", "small", "square", "wide169", "wide43", "tall34", "tall916"])
    func network(name: String) async throws {
        let estimator = try await DepthEstimator(contentsOf: installed)
        let m = try manifest()
        let image = try #require(m.images[name])
        let size = DepthEstimator.size(fromName: image.input)!
        let predicted = try await estimator.predict(DepthInput(rgb: try rgb(name, image), width: image.width, height: image.height, size: size))
        let reference = try floats("\(name)_depth")
        let db = psnr(reference, predicted)
        print("network \(name) \(image.input): \(String(format: "%.1f", db)) dB over \(predicted.count) values (range \(image.range[0])–\(image.range[1]))")
        #expect(db > 60)
    }

    @Test("whole images match infer_image, upsampling included", .enabled(if: modelInstalled), arguments: ["street", "small", "square", "tall916"])
    func endToEnd(name: String) async throws {
        let estimator = try await DepthEstimator(contentsOf: installed)
        let m = try manifest()
        let image = try #require(m.images[name])
        let (map, exact) = try await estimator.depth(rgb: try rgb(name, image), width: image.width, height: image.height)
        #expect(exact && map.inputSize.name == image.input)
        let reference = try floats("\(name)_full")
        let db = psnr(reference, map.values)
        print("end-to-end \(name): \(String(format: "%.1f", db)) dB at \(map.width)x\(map.height)")
        #expect(db > 60)
        #expect(map.image8() != nil && map.image16() != nil)
    }

    @Test("an unexported aspect ratio falls back to the nearest exported size", .enabled(if: modelInstalled))
    func fallback() async throws {
        let estimator = try await DepthEstimator(contentsOf: installed)
        let (size, exact) = estimator.size(for: 2000, 500)          // 4:1 → nearest exported is 16:9 landscape
        #expect(!exact && size.name == "h518w924")
        #expect(estimator.size(for: 1600, 1200).exact)
    }
}
