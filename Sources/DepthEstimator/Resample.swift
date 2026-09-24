//
//  Resample.swift
//  DepthEstimator
//
//  Created by David Sherlock on 2026.
//
//  Two resamplers the reference pipeline uses, reproduced exactly:
//  OpenCV's INTER_CUBIC (the Catmull-Rom-style kernel with A = −0.75,
//  no antialiasing, half-pixel centres, replicated borders, coefficients in
//  float, sums in double for a double image) for the input, and PyTorch's
//  bilinear `interpolate` with `align_corners=True` for the output.
//

import Foundation

public enum Resample {

    /// OpenCV's `interpolateCubic` with A = −0.75.
    @inline(__always)
    static func cubicCoefficients(_ x: Float) -> (Float, Float, Float, Float) {
        let A: Float = -0.75
        let c0 = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A
        let c1 = ((A + 2) * x - (A + 3)) * x * x + 1
        let c2 = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1
        return (c0, c1, c2, 1 - c0 - c1 - c2)
    }

    /// `cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)` for a double image
    /// laid out `[height][width][channels]`: horizontal pass then vertical, each source position
    /// `(dst + 0.5) · scale − 0.5`, taps clamped to the edges.
    public static func cubic(_ src: [Double], width sw: Int, height sh: Int, channels: Int, toWidth dw: Int, toHeight dh: Int) -> [Double] {
        let scaleX = Double(sw) / Double(dw), scaleY = Double(sh) / Double(dh)
        // Horizontal taps and weights per destination column.
        var xIndex = [[Int]](repeating: [0, 0, 0, 0], count: dw)
        var xWeight = [(Float, Float, Float, Float)](repeating: (0, 0, 0, 0), count: dw)
        for dx in 0..<dw {
            let fx = (Double(dx) + 0.5) * scaleX - 0.5
            var sx = Int(floor(fx))
            let frac = Float(fx - Double(sx))
            sx -= 1
            xIndex[dx] = (0..<4).map { min(max(sx + $0, 0), sw - 1) }
            xWeight[dx] = cubicCoefficients(frac)
        }
        var yIndex = [[Int]](repeating: [0, 0, 0, 0], count: dh)
        var yWeight = [(Float, Float, Float, Float)](repeating: (0, 0, 0, 0), count: dh)
        for dy in 0..<dh {
            let fy = (Double(dy) + 0.5) * scaleY - 0.5
            var sy = Int(floor(fy))
            let frac = Float(fy - Double(sy))
            sy -= 1
            yIndex[dy] = (0..<4).map { min(max(sy + $0, 0), sh - 1) }
            yWeight[dy] = cubicCoefficients(frac)
        }
        // Horizontal pass over every source row that any destination row touches: all of them, in practice.
        var rows = [Double](repeating: 0, count: sh * dw * channels)
        for sy in 0..<sh {
            let srow = sy * sw * channels
            for dx in 0..<dw {
                let (w0, w1, w2, w3) = xWeight[dx]
                let i = xIndex[dx]
                for c in 0..<channels {
                    rows[(sy * dw + dx) * channels + c] = src[srow + i[0] * channels + c] * Double(w0) + src[srow + i[1] * channels + c] * Double(w1)
                        + src[srow + i[2] * channels + c] * Double(w2) + src[srow + i[3] * channels + c] * Double(w3)
                }
            }
        }
        var out = [Double](repeating: 0, count: dh * dw * channels)
        for dy in 0..<dh {
            let (w0, w1, w2, w3) = yWeight[dy]
            let r = yIndex[dy]
            for dx in 0..<dw {
                for c in 0..<channels {
                    let k = dx * channels + c
                    out[(dy * dw + dx) * channels + c] = rows[(r[0] * dw) * channels + k] * Double(w0) + rows[(r[1] * dw) * channels + k] * Double(w1)
                        + rows[(r[2] * dw) * channels + k] * Double(w2) + rows[(r[3] * dw) * channels + k] * Double(w3)
                }
            }
        }
        return out
    }

    /// `torch.nn.functional.interpolate(x, (height, width), mode="bilinear", align_corners=True)`
    /// for one channel: source position `dst · (in − 1) / (out − 1)`, float arithmetic.
    public static func bilinearAlignCorners(_ src: [Float], width sw: Int, height sh: Int, toWidth dw: Int, toHeight dh: Int) -> [Float] {
        if sw == dw && sh == dh { return src }
        let rw: Float = dw > 1 ? Float(sw - 1) / Float(dw - 1) : 0
        let rh: Float = dh > 1 ? Float(sh - 1) / Float(dh - 1) : 0
        var x0 = [Int](repeating: 0, count: dw), x1 = [Int](repeating: 0, count: dw), lx = [Float](repeating: 0, count: dw)
        for dx in 0..<dw {
            let s = rw * Float(dx)
            let i = min(Int(s), sw - 1)
            x0[dx] = i; x1[dx] = min(i + 1, sw - 1); lx[dx] = s - Float(i)
        }
        var out = [Float](repeating: 0, count: dw * dh)
        for dy in 0..<dh {
            let s = rh * Float(dy)
            let j = min(Int(s), sh - 1), j1 = min(j + 1, sh - 1)
            let ly = s - Float(j)
            for dx in 0..<dw {
                let a = src[j * sw + x0[dx]], b = src[j * sw + x1[dx]], c = src[j1 * sw + x0[dx]], d = src[j1 * sw + x1[dx]]
                let top = a + (b - a) * lx[dx], bottom = c + (d - c) * lx[dx]
                out[dy * dw + dx] = top + (bottom - top) * ly
            }
        }
        return out
    }
}
