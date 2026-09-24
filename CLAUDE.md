# CLAUDE.md — swift-depth-estimator

Depth Anything V2 Small on Core AI: `DepthEstimator` (nine fixed-size entry
points `h<H>w<W>`, picks the authors' size or the nearest), `DepthInput`
(size rule + exact OpenCV cubic resize + ImageNet normalisation),
`Resample` (cv2 INTER_CUBIC, torch bilinear align_corners), `DepthMap`.
Module `DepthEstimator`; the verb is `img depth` in `../swift-img-cli`.
Added 2026-09-24.

## Build & test
```bash
swift build && swift test        # size rule, tensors and resamplers need no model
```
Model-backed tests need `~/Library/Application Support/img/img-depth-anything-v2-small-float32.aimodel`:
`uv run Tools/export_depth_anything.py --repo <clone> --checkpoint depth_anything_v2_vits.pth --install name=img.jpg …`
(clone of github.com/DepthAnything/Depth-Anything-V2 in `scratchpad/depth/`,
checkpoint from HF `depth-anything/Depth-Anything-V2-Small`; probe venv with
torch 2.13 + torchvision 0.28 + opencv-python-headless).

## Traps (all hit)
- **The reference is the authors' repo, not transformers.** transformers 5's
  image processor needs torchvision and resizes with torchvision (antialiased
  bicubic) then upsamples bicubic; the paper's `infer_image` uses OpenCV
  INTER_CUBIC (no antialias) and bilinear `align_corners=True`. Different
  numbers. `image2tensor` also moves the tensor to MPS by itself — patch
  `torch.backends.mps.is_available` to False for CPU references.
- **`np.round` is half-to-even** in `constrain_to_multiple_of`; Swift's
  `.rounded()` is not. 64.5 → 64 vs 65.
- **A multi-entry export shares weights**: 9 sizes = 102 MB, not 9 × 100.
- **torchvision pulls torch forward** (2.14) and coreai-torch is validated on
  ≤ 2.13 — pin `torch==2.13.0 torchvision==0.28.0`.
- A cropped fixture from a source that is too short gave a 766×15 image →
  a 518×25900 tensor → the export was SIGKILLed with no traceback.
- Fixture `.rgb` files are raw 8-bit RGB; sizes are in `manifest.json`.
