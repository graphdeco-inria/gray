"""
Convert a GRay scene into an INRIA 3D Gaussian Splatting model directory.

A GRay scene directory contains:
    config.json                  training configuration
    cameras.json                 GRay camera list (R, T, origin, fov_x/y, ...)
    gaussians_<iter>.safetensors trained gaussians

This produces the 3DGS layout consumed by render.py, the SIBR viewer,
graphdecoviewer and online viewers (e.g. supersplat):
    <out>/cfg_args                                   argparse.Namespace repr
    <out>/cameras.json                               3DGS camera list
    <out>/point_cloud/iteration_<iter>/point_cloud.ply

The strict minimum to *view* the splat (supersplat / graphdecoviewer) is the
point_cloud.ply alone. The SIBR viewer and render.py additionally need cfg_args
(for source_path / white_background / sh_degree) and cameras.json (for the
initial camera poses), so we always emit the full bundle.
"""

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import re
from typing import Optional

import numpy as np
import tyro
from safetensors import safe_open
from tyro.conf import Positional

from safetensors_ply_conversion import safetensors_to_ply


@dataclass
class ConversionCLI:
    # * GRay scene directory (config.json, cameras.json, gaussians_*.safetensors)
    input_path: Positional[str]
    # * 3DGS model directory to create (default: <input>/3dgs)
    output_path: Positional[Optional[str]] = None
    # * Which gaussians_<iter>.safetensors to convert (default: highest iteration)
    iteration: Optional[int] = None
    # * Override the COLMAP source_path written to cfg_args (default: from config.json)
    source_path: Optional[str] = None
    # * data_device written to cfg_args
    data_device: str = "cuda"


cli = tyro.cli(ConversionCLI)
scene_dir = Path(cli.input_path)
if not scene_dir.is_dir():
    raise NotADirectoryError(f"Input must be a GRay scene directory: {scene_dir}")

out_dir = Path(cli.output_path) if cli.output_path else scene_dir / "3dgs"
out_dir.mkdir(parents=True, exist_ok=True)

# * Locate the gaussians safetensors (highest iteration unless one was requested)
candidates = {}
for path in scene_dir.glob("gaussians_*.safetensors"):
    match = re.fullmatch(r"gaussians_(\d+)", path.stem)
    if match:
        candidates[int(match.group(1))] = path
if not candidates:
    raise FileNotFoundError(f"No gaussians_<iter>.safetensors found in {scene_dir}")
iteration = cli.iteration if cli.iteration is not None else max(candidates)
if iteration not in candidates:
    raise FileNotFoundError(f"gaussians_{iteration}.safetensors not found (have {sorted(candidates)})")

# * Write the 3DGS point_cloud.ply using the shared safetensors<->ply conversion
ply_path = out_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
ply_path.parent.mkdir(parents=True, exist_ok=True)
safetensors_to_ply(candidates[iteration], ply_path)

# * cfg_args needs the SH degree, and it must match the ply's f_rest count or 3DGS
# * load_ply asserts; read it straight from the header (15 rest coeffs => degree 3)
with safe_open(str(candidates[iteration]), framework="pt") as f:
    if "sh_coeffs_rest" in f.keys():
        sh_degree = math.isqrt(f.get_slice("sh_coeffs_rest").get_shape()[1] + 1) - 1
    else:
        sh_degree = 0
print(f"  point_cloud -> {ply_path}  (sh_degree={sh_degree})")

# * config.json -> cfg_args (the argparse.Namespace repr that 3DGS eval()s)
config_path = scene_dir / "config.json"
config = json.loads(config_path.read_text()) if config_path.exists() else {}
if cli.source_path is not None:
    source_path = os.path.abspath(cli.source_path)
else:
    raw = config.get("source_path", "")
    # * GRay paths are relative to the repo root (this script lives in convert/ under it)
    repo_root = Path(__file__).resolve().parent.parent
    source_path = raw if os.path.isabs(raw) else str((repo_root / raw).resolve())
match = re.search(r"images_(\d+)", str(config.get("images_dir") or "images"))
resolution = int(match.group(1)) if match else 1
bg = config.get("bg_color", [0.0, 0.0, 0.0])
white_background = len(bg) == 3 and all(abs(c - 1.0) < 1e-6 for c in bg)
cfg_args = repr(
    Namespace(
        sh_degree=sh_degree,
        source_path=source_path,
        model_path=str(out_dir) + os.sep,
        images="images",  # * base folder; render.py selects images_<resolution> from -r
        depths="",
        pc="point_cloud.safetensors",
        resolution=resolution,
        white_background=white_background,
        train_test_exp=False,
        data_device=cli.data_device,
        eval=bool(config.get("eval", True)),
    )
)
(out_dir / "cfg_args").write_text(cfg_args)
print(f"  cfg_args    -> {out_dir / 'cfg_args'}")
print(f"              {cfg_args}")

# * cameras.json: GRay {uid,R,origin,fov_*} -> 3DGS {id,rotation,position,fx,fy}
cameras_path = scene_dir / "cameras.json"
if cameras_path.exists():
    cameras = []
    for idx, cam in enumerate(json.loads(cameras_path.read_text())):
        width, height = int(cam["image_width"]), int(cam["image_height"])
        cameras.append(
            {
                "id": int(cam.get("uid", idx)),
                "img_name": cam.get("image_name", f"{idx:05d}"),
                "width": width,
                "height": height,
                "position": np.asarray(cam["origin"], dtype=np.float64).tolist(),  # * camera center
                "rotation": np.asarray(cam["R"], dtype=np.float64).tolist(),  # * camera-to-world
                "fy": height / (2.0 * math.tan(float(cam["fov_y"]) / 2.0)),
                "fx": width / (2.0 * math.tan(float(cam["fov_x"]) / 2.0)),
            }
        )
    (out_dir / "cameras.json").write_text(json.dumps(cameras))
    print(f"  cameras.json-> {out_dir / 'cameras.json'}  ({len(cameras)} cameras)")
else:
    print(f"  WARNING: no cameras.json in {scene_dir}; SIBR viewer needs one")

print(f"Converted GRay scene -> 3DGS model at {out_dir}")
