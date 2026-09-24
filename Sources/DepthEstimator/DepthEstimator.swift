//
//  DepthEstimator.swift
//  DepthEstimator
//
//  Created by David Sherlock on 2026.
//
//  Depth Anything V2 Small (Yang et al., NeurIPS 2024; Apache 2.0) on Core
//  AI. The network — DINOv2-small and the DPT head — sits in the .aimodel
//  at a set of fixed input sizes, one entry point each, because the
//  authors' `Resize` picks the size from the image's aspect ratio and the
//  runtime wants static shapes. Everything around it is the authors'
//  `infer_image`: OpenCV cubic resize, ImageNet normalisation, bilinear
//  upsampling with aligned corners back to the image's own size.
//

import CoreAI
import CoreGraphics
import Foundation

public final class DepthEstimator: @unchecked Sendable {

    public static let assetName = "img-depth-anything-v2-small-float32.aimodel"

    public let url: URL
    /// The input sizes the asset holds, from its entry point names.
    public let sizes: [DepthInput.Size]
    private var functions: [DepthInput.Size: InferenceFunction] = [:]

    public init(contentsOf url: URL) async throws {
        guard FileManager.default.fileExists(atPath: url.path) else { throw DepthEstimatorError.modelNotFound(url.path) }
        self.url = url
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model: AIModel
        do { model = try await AIModel(contentsOf: url, options: options) } catch {
            throw DepthEstimatorError.modelUnavailable("could not load \(url.lastPathComponent): \(error)")
        }
        var sizes: [DepthInput.Size] = []
        for name in model.functionNames {
            guard let size = Self.size(fromName: name), let f = try model.loadFunction(named: name) else { continue }
            functions[size] = f; sizes.append(size)
        }
        guard !sizes.isEmpty else { throw DepthEstimatorError.modelUnavailable("\(url.lastPathComponent) has no h<height>w<width> entry points; found \(model.functionNames)") }
        self.sizes = sizes.sorted { ($0.height, $0.width) < ($1.height, $1.width) }
    }

    /// "h518w686" → 686 × 518.
    static func size(fromName name: String) -> DepthInput.Size? {
        guard name.hasPrefix("h"), let w = name.firstIndex(of: "w"), let height = Int(name[name.index(after: name.startIndex)..<w]),
              let width = Int(name[name.index(after: w)...]) else { return nil }
        return DepthInput.Size(width: width, height: height)
    }

    /// The exported size an image of `width × height` should use: the authors' own choice when
    /// the asset has it, else the exported size whose aspect ratio is nearest (a small distortion
    /// the model tolerates; `exact` says which happened).
    public func size(for width: Int, _ height: Int) -> (size: DepthInput.Size, exact: Bool) {
        let wanted = DepthInput.Size.reference(for: width, height)
        if functions[wanted] != nil { return (wanted, true) }
        let ratio = log(Double(width) / Double(height))
        let nearest = sizes.min { abs(log(Double($0.width) / Double($0.height)) - ratio) < abs(log(Double($1.width) / Double($1.height)) - ratio) }!
        return (nearest, false)
    }

    /// The network on a prepared input: `[height][width]` relative inverse depth at the input size.
    public func predict(_ input: DepthInput) async throws -> [Float] {
        guard let function = functions[input.size] else { throw DepthEstimatorError.sizeUnsupported("no entry point for \(input.size.name); the asset has \(sizes.map(\.name).joined(separator: " "))") }
        var out = NDArray(shape: [1, input.size.height, input.size.width], scalarType: .float32)
        var views = InferenceFunction.MutableViews()
        views.insert(out.mutableRawView(), for: "depth")
        do {
            _ = try await function.run(inputs: ["image": Self.array(input.tensor, shape: [1, 3, input.size.height, input.size.width])],
                                       states: InferenceFunction.MutableViews(), outputViews: consume views)
        } catch { throw DepthEstimatorError.modelUnavailable("inference failed: \(error)") }
        return Self.floats(out)
    }

    /// `infer_image` for 8-bit RGB pixels: the depth map at the image's own size.
    public func depth(rgb: [UInt8], width: Int, height: Int) async throws -> (map: DepthMap, exact: Bool) {
        let (size, exact) = self.size(for: width, height)
        let input = DepthInput(rgb: rgb, width: width, height: height, size: size)
        let predicted = try await predict(input)
        let full = Resample.bilinearAlignCorners(predicted, width: size.width, height: size.height, toWidth: width, toHeight: height)
        return (DepthMap(width: width, height: height, values: full, inputSize: size), exact)
    }

    public func depth(of image: CGImage) async throws -> (map: DepthMap, exact: Bool) {
        guard let rgb = DepthInput.rgb(of: image) else { throw DepthEstimatorError.imageUnreadable("CGImage") }
        return try await depth(rgb: rgb.pixels, width: rgb.width, height: rgb.height)
    }

    public func depth(contentsOf url: URL) async throws -> (map: DepthMap, exact: Bool) {
        let rgb = try DepthInput.rgb(contentsOf: url)
        return try await depth(rgb: rgb.pixels, width: rgb.width, height: rgb.height)
    }

    static func array(_ values: [Float], shape: [Int]) -> NDArray {
        var a = NDArray(shape: shape, scalarType: .float32)
        let view = a.mutableView(as: Float.self)
        view.withUnsafeMutablePointer { p, _, _ in values.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: values.count) } }
        return a
    }

    static func floats(_ array: NDArray) -> [Float] {
        let count = array.shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        array.view(as: Float.self).withUnsafePointer { p, _, _ in out.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: p, count: count) } }
        return out
    }
}
