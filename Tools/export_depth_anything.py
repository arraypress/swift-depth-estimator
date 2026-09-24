# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.13.0", "torchvision==0.28.0", "numpy", "coreai-torch==0.4.2", "coreai-core==1.0.0b2", "opencv-python-headless"]
# ///
"""Depth Anything V2 Small (Yang et al., NeurIPS 2024; Apache 2.0) → img-depth-anything-v2-small-float32.aimodel + fixtures.

    uv run Tools/export_depth_anything.py --repo <clone of DepthAnything/Depth-Anything-V2> --checkpoint depth_anything_v2_vits.pth \\
        [--sizes 518x518,686x518,…] [--out Tools/exports] [--install] [name=image.jpg …]

The network is the authors' own `DepthAnythingV2(encoder="vits")` with the official checkpoint; nothing from a
conversion. One entry point per fixed input size, named h<height>w<width>, because the authors' `Resize` picks
the size from the aspect ratio (`lower_bound`: shorter side 518, both sides multiples of 14) and the runtime wants
static shapes. Everything around the network in the host is `infer_image` line for line: OpenCV INTER_CUBIC on the
double image in 0…1, ImageNet normalisation, bilinear `align_corners=True` back to the image's own size.
Fixtures per image: the 8-bit RGB (as .rgb), the prepared tensor, the predicted depth, the upsampled depth."""
import argparse, os, sys, json, time, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, cv2
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--checkpoint", required=True)
ap.add_argument("--sizes", default="518x518,686x518,518x686,784x518,518x784,924x518,518x924,826x518,518x826", help="WxH, comma-separated")
ap.add_argument("--asset", default="img-depth-anything-v2-small-float32.aimodel"); ap.add_argument("--out", default=str(HERE / "exports"))
ap.add_argument("--fixtures", default=str(HERE.parent / "Tests/DepthEstimatorTests/Fixtures")); ap.add_argument("--install", action="store_true")
ap.add_argument("--max-side", type=int, default=800, help="fixture images are shrunk to this on the long side first")
ap.add_argument("images", nargs="*", help="name=path.jpg")
args = ap.parse_args()
sys.path.insert(0, args.repo)
torch.backends.mps.is_available = lambda: False
from depth_anything_v2.dpt import DepthAnythingV2
import coreai_torch
from coreai.runtime import AIModelAssetMetadata
model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")); model.eval()
sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
print(f"Depth Anything V2 Small: {sum(p.numel() for p in model.parameters())/1e6:.1f} M parameters; sizes {sizes}", flush=True)

fx = Path(args.fixtures); fx.mkdir(parents=True, exist_ok=True)
manifest = {"sizes": [f"h{h}w{w}" for w, h in sizes], "images": {}}
with torch.no_grad():
    for spec in args.images:
        name, path = spec.split("=", 1)
        bgr = cv2.imread(path)
        h0, w0 = bgr.shape[:2]
        if max(h0, w0) > args.max_side:
            s = args.max_side / max(h0, w0)
            bgr = cv2.resize(bgr, (int(round(w0 * s)), int(round(h0 * s))), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x, (hh, ww) = model.image2tensor(bgr, 518)               # the authors' preprocessing, on the CPU
        depth = model.forward(x)                                  # [1, H', W']
        full = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
        H, W = x.shape[2], x.shape[3]
        rgb.astype(np.uint8).tofile(fx / f"{name}.rgb")
        x[0].numpy().astype(np.float32).tofile(fx / f"{name}_input.f32")
        depth[0].numpy().astype(np.float32).tofile(fx / f"{name}_depth.f32")
        full.numpy().astype(np.float32).tofile(fx / f"{name}_full.f32")
        manifest["images"][name] = {"width": w, "height": h, "input": f"h{H}w{W}", "exported": (W, H) in sizes,
                                    "range": [float(depth.min()), float(depth.max())]}
        print(f"{name}: {w}x{h} → input h{H}w{W} ({'exported' if (W, H) in sizes else 'NOT an exported size'}); depth {float(depth.min()):.2f}–{float(depth.max()):.2f}", flush=True)
json.dump(manifest, open(fx / "manifest.json", "w"), indent=1)

class Net(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, image): return self.m(image)
net = Net(model).eval()
t0 = time.time()
converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
with torch.no_grad():
    for w, h in sizes:
        ex = torch.export.export(net, (torch.randn(1, 3, h, w),)).run_decompositions(coreai_torch.get_decomp_table())
        converter.add_exported_program(ex, input_names=["image"], output_names=["depth"], entrypoint_name=f"h{h}w{w}")
        print(f"exported h{h}w{w} in {time.time()-t0:.0f} s", flush=True)
program = converter.to_coreai(); program.optimize()
meta = AIModelAssetMetadata()
meta.author = "Yang, Kang, Ming, Xu, Feng, Zhao — Depth Anything V2 (NeurIPS 2024), Small; Core AI export by DepthEstimator"
meta.license = "Apache-2.0"
meta.model_description = f"Depth Anything V2 Small: entry points {', '.join(f'h{h}w{w}' for w, h in sizes)} take an ImageNet-normalised RGB image [1,3,H,W] and return relative inverse depth [1,H,W]. Preprocess and upsample in the host as the authors' infer_image does."
meta.creation_date = int(time.time())
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
asset = out / args.asset
if asset.exists(): shutil.rmtree(asset)
program.save_asset(asset, meta)
print(f"saved {asset} ({sum(f.stat().st_size for f in asset.rglob('*'))/1e6:.0f} MB) in {time.time()-t0:.0f} s", flush=True)
if args.install:
    dest = Path.home() / "Library/Application Support/img"; dest.mkdir(parents=True, exist_ok=True)
    d = dest / asset.name
    if d.exists(): shutil.rmtree(d)
    shutil.copytree(asset, d); print(f"installed into {dest}")
