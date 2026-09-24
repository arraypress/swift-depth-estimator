//
//  DepthInput.swift
//  DepthEstimator
//
//  Created by David Sherlock on 2026.
//
//  The reference pipeline's `image2tensor`: an 8-bit RGB image becomes a
//  double image in 0…1, is resized with OpenCV's cubic filter so its
//  shorter side is 518 and both sides are multiples of 14 (`lower_bound`),
//  normalised with ImageNet's mean and deviation, and laid out CHW as
//  float32. `Size.reference(for:)` is the size the authors' code would
//  choose; `Size.buckets` are the sizes the asset was exported at.
//

import CoreGraphics
import Foundation
import ImageIO

public struct DepthInput: Sendable {

    public struct Size: Hashable, Sendable {
        public let width: Int
        public let height: Int
        public init(width: Int, height: Int) { self.width = width; self.height = height }
        public var name: String { "h\(height)w\(width)" }
        /// The reference `Resize(518, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method="lower_bound")`.
        public static func reference(for width: Int, _ height: Int, target: Int = 518, multiple: Int = 14) -> Size {
            let scaleW = Double(target) / Double(width), scaleH = Double(target) / Double(height)
            let scale = max(scaleW, scaleH)      // lower_bound: the smaller side reaches the target
            func constrain(_ x: Double) -> Int {
                var y = Int((x / Double(multiple)).rounded(.toNearestOrEven)) * multiple      // np.round: half to even
                if y < target { y = Int(ceil(x / Double(multiple))) * multiple }
                return y
            }
            return Size(width: constrain(scale * Double(width)), height: constrain(scale * Double(height)))
        }
    }

    public static let mean: [Double] = [0.485, 0.456, 0.406]
    public static let std: [Double] = [0.229, 0.224, 0.225]

    public let size: Size
    /// `[3][height][width]` float32, normalised.
    public let tensor: [Float]

    /// An 8-bit RGB image (`[height][width][3]`) through the reference pipeline at `size`.
    public init(rgb: [UInt8], width: Int, height: Int, size: Size) {
        var image = [Double](repeating: 0, count: rgb.count)
        for i in rgb.indices { image[i] = Double(rgb[i]) / 255.0 }
        let resized = Resample.cubic(image, width: width, height: height, channels: 3, toWidth: size.width, toHeight: size.height)
        var tensor = [Float](repeating: 0, count: 3 * size.width * size.height)
        let plane = size.width * size.height
        for p in 0..<plane {
            for c in 0..<3 { tensor[c * plane + p] = Float((resized[p * 3 + c] - Self.mean[c]) / Self.std[c]) }
        }
        self.size = size
        self.tensor = tensor
    }

    /// The 8-bit RGB pixels of a CGImage, row-major, alpha dropped, orientation as decoded.
    public static func rgb(of image: CGImage) -> (pixels: [UInt8], width: Int, height: Int)? {
        let width = image.width, height = image.height
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: &rgba, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: space,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return nil }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for p in 0..<(width * height) { rgb[p * 3] = rgba[p * 4]; rgb[p * 3 + 1] = rgba[p * 4 + 1]; rgb[p * 3 + 2] = rgba[p * 4 + 2] }
        return (rgb, width, height)
    }

    /// The first image in a file, orientation applied.
    public static func rgb(contentsOf url: URL) throws -> (pixels: [UInt8], width: Int, height: Int) {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary) else {
            throw DepthEstimatorError.imageUnreadable(url.path)
        }
        let oriented = Self.applyingOrientation(image, from: source)
        guard let rgb = rgb(of: oriented) else { throw DepthEstimatorError.imageUnreadable(url.path) }
        return rgb
    }

    static func applyingOrientation(_ image: CGImage, from source: CGImageSource) -> CGImage {
        guard let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let raw = props[kCGImagePropertyOrientation] as? UInt32, let orientation = CGImagePropertyOrientation(rawValue: raw), orientation != .up else { return image }
        let w = image.width, h = image.height
        let swaps = [CGImagePropertyOrientation.left, .right, .leftMirrored, .rightMirrored].contains(orientation)
        let (ow, oh) = swaps ? (h, w) : (w, h)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: ow, height: oh, bitsPerComponent: 8, bytesPerRow: 0, space: space,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return image }
        context.translateBy(x: CGFloat(ow) / 2, y: CGFloat(oh) / 2)
        switch orientation {
        case .down, .downMirrored: context.rotate(by: .pi)
        case .left, .leftMirrored: context.rotate(by: .pi / 2)
        case .right, .rightMirrored: context.rotate(by: -.pi / 2)
        default: break
        }
        if [CGImagePropertyOrientation.upMirrored, .downMirrored, .leftMirrored, .rightMirrored].contains(orientation) { context.scaleBy(x: -1, y: 1) }
        context.draw(image, in: CGRect(x: -CGFloat(w) / 2, y: -CGFloat(h) / 2, width: CGFloat(w), height: CGFloat(h)))
        return context.makeImage() ?? image
    }
}

public enum DepthEstimatorError: Error, CustomStringConvertible {
    case modelNotFound(String), modelUnavailable(String), imageUnreadable(String), sizeUnsupported(String)
    public var description: String {
        switch self {
        case .modelNotFound(let p): return "no model at \(p)"
        case .modelUnavailable(let m): return m
        case .imageUnreadable(let p): return "could not read \(p) as an image"
        case .sizeUnsupported(let m): return m
        }
    }
}
