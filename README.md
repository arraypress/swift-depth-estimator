# DepthEstimator

On-device **monocular depth** for Swift — one photograph in, a depth map out, nearer is
brighter. Depth Anything V2 Small (Yang et al., NeurIPS 2024; Apache 2.0) runs on Core AI; the
authors' preprocessing, the exact OpenCV cubic resampler and the bilinear upsampling back to the
picture's size are Swift. Nothing leaves the machine.

```swift
import DepthEstimator

let estimator = try await DepthEstimator(contentsOf: assetURL)   // img-depth-anything-v2-small-float32.aimodel
let (map, exact) = try await estimator.depth(contentsOf: photoURL)
map.values                 // [height][width] relative inverse depth, larger is nearer
map.normalised()           // 0…1, nearest = 1
map.image16()              // a 16-bit greyscale CGImage, or image8()
```

## What it is exactly

The network is the authors' own `DepthAnythingV2(encoder: "vits")` — DINOv2-small with the DPT
head, 24.8 M parameters — loaded from the official checkpoint and exported at nine fixed input
sizes as entry points `h<height>w<width>`: the sizes their `Resize` (shorter side 518, both sides
multiples of 14, `lower_bound`) produces for square, 4:3, 3:2, 16:9 and 16:10 pictures in both
orientations. The weights are shared, so the asset is 102 MB. `DepthInput.Size.reference(for:)`
is that rule, `np.round`'s half-to-even included; a picture whose ratio falls between the
exported sizes uses the nearest one, which `exact` reports.

Everything else is `infer_image` line for line. `Resample.cubic` is OpenCV's `INTER_CUBIC` — the
A = −0.75 kernel, half-pixel centres, replicated edges, float coefficients and double sums — on
the image in 0…1; then ImageNet normalisation and a float32 CHW tensor (`DepthInput`); then the
network; then `Resample.bilinearAlignCorners`, PyTorch's `interpolate(mode: bilinear,
align_corners: true)`, to the picture's own size (`DepthMap`).

## Measured against upstream

`Tools/export_depth_anything.py` runs the authors' code on their demo images (cropped to every
exported ratio, shrunk to 800 px) and writes the 8-bit pixels, the prepared tensor, the network's
output and the upsampled map; `swift test` compares.

| Stage | Agreement with the authors' PyTorch (CPU) |
|---|---|
| Size rule, 7 fixtures + 8 probe cases | identical |
| Prepared tensor (cubic resize + normalise), 5 pictures | > 100 dB PSNR |
| Network on Core AI, 7 pictures at 7 sizes | 111–127 dB |
| Whole pictures end to end, 4 | 113–130 dB |

On an M3 Max, release build, a 2048 × 1362 photograph takes 0.19 s from pixels to map and a
640 × 425 one 0.07 s; loading the asset adds about 0.3 s to a process.

## The model

Not bundled (102 MB). Export it from the authors' repo and checkpoint, or download the export:

```sh
uv run Tools/export_depth_anything.py --repo Depth-Anything-V2 --checkpoint depth_anything_v2_vits.pth --install
hf download arraypress/img-depth-anything --local-dir models
```

The command-line tool is [`img depth`](https://github.com/arraypress/swift-img-cli).

## Requirements

- macOS 27+ (Core AI), Apple silicon
- Swift 6

## License

MIT — see [LICENSE](LICENSE). Depth Anything V2 Small and its weights are Apache 2.0 (the Base
and Large checkpoints are CC BY-NC 4.0 and are not used).
