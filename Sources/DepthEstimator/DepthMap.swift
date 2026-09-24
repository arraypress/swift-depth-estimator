//
//  DepthMap.swift
//  DepthEstimator
//
//  Created by David Sherlock on 2026.
//
//  What the model returns: relative inverse depth — larger is nearer — one
//  value per pixel of the original image, plus the ways it is usually
//  read: normalised to 0…1 or 0…255, near-to-far order preserved.
//

import CoreGraphics
import Foundation

public struct DepthMap: Sendable {

    public let width: Int
    public let height: Int
    /// `[height][width]` relative inverse depth as the model produced it (after upsampling).
    public let values: [Float]
    /// The model input size used.
    public let inputSize: DepthInput.Size

    public var minimum: Float { values.min() ?? 0 }
    public var maximum: Float { values.max() ?? 0 }

    /// `(value − min) / (max − min)`: 1 is nearest.
    public func normalised() -> [Float] {
        let lo = minimum, hi = maximum
        let span = hi - lo
        return span > 0 ? values.map { ($0 - lo) / span } : values.map { _ in 0 }
    }

    /// An 8-bit greyscale image, nearest = white (the authors' `depth * 255` after normalising).
    public func image8() -> CGImage? {
        let bytes = normalised().map { UInt8(min(255, max(0, ($0 * 255).rounded()))) }
        return Self.grey(bytes, width: width, height: height, bits: 8)
    }

    /// A 16-bit greyscale image, nearest = 65535, for editors that keep the precision.
    public func image16() -> CGImage? {
        let n = normalised()
        var bytes = [UInt8](repeating: 0, count: n.count * 2)
        for (i, v) in n.enumerated() {
            let s = UInt16(min(65535, max(0, (v * 65535).rounded())))
            bytes[i * 2] = UInt8(s >> 8); bytes[i * 2 + 1] = UInt8(s & 0xFF)
        }
        return Self.grey(bytes, width: width, height: height, bits: 16)
    }

    static func grey(_ bytes: [UInt8], width: Int, height: Int, bits: Int) -> CGImage? {
        let data = Data(bytes)
        guard let provider = CGDataProvider(data: data as CFData), let space = CGColorSpace(name: CGColorSpace.linearGray) ?? CGColorSpace(name: CGColorSpace.genericGrayGamma2_2) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: bits, bitsPerPixel: bits, bytesPerRow: width * bits / 8, space: space,
                       bitmapInfo: bits == 16 ? CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder16Big.rawValue) : CGBitmapInfo(rawValue: 0),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
