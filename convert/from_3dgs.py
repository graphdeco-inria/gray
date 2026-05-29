"""
Convert an INRIA 3D Gaussian Splatting model directory into a GRay scene.

Reverse of to_3dgs.py. Reads:
    <in>/cfg_args                                    argparse.Namespace repr
    <in>/cameras.json                                3DGS camera list
    <in>/point_cloud/iteration_<iter>/point_cloud.ply

and writes a GRay scene directory:
    <out>/config.json
    <out>/cameras.json
    <out>/gaussians_<iter>.safetensors

The PLY -> safetensors step reuses safetensors_ply_conversion.py, which reads
gaussian attributes by name (so the extra nx/ny/nz normals are ignored).
config.json is rebuilt from cfg_args on top of GRay's defaults; GRay training
hyper-parameters that 3DGS does not store fall back to those defaults.
"""

from argparse import Namespace 
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os
import re
import sys
from typing import Optional

import numpy as np
import tyro
from tyro.conf import Positional

from safetensors_ply_conversion import ply_to_safetensors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gray.config import Config


@dataclass
class ConversionCLI:
    # * 3DGS model directory (cfg_args + point_cloud/iteration_*/point_cloud.ply)
    input_path: Positional[str]
    # * GRay scene directory to create (default: <input>/gray)
    output_path: Positional[Optional[str]] = None
    # * Which point_cloud/iteration_<iter> to convert (default: highest iteration)
    iteration: Optional[int] = None


cli = tyro.cli(ConversionCLI)
model_dir = Path(cli.input_path)
if not model_dir.is_dir():
    raise NotADirectoryError(f"Input must be a 3DGS model directory: {model_dir}")

out_dir = Path(cli.output_path) if cli.output_path else model_dir / "gray"
out_dir.mkdir(parents=True, exist_ok=True)

# * Locate the point cloud (highest iteration unless one was requested)
candidates = {}
for path in (model_dir / "point_cloud").glob("iteration_*"):
    match = re.fullmatch(r"iteration_(\d+)", path.name)
    if match and (path / "point_cloud.ply").exists():
        candidates[int(match.group(1))] = path / "point_cloud.ply"
if not candidates:
    raise FileNotFoundError(f"No point_cloud/iteration_*/point_cloud.ply under {model_dir}")
iteration = cli.iteration if cli.iteration is not None else max(candidates)
if iteration not in candidates:
    raise FileNotFoundError(f"iteration_{iteration} not found (have {sorted(candidates)})")

# * PLY -> GRay safetensors (shared code reads gaussian attributes by name);
# * GRay loads gaussians_<iter> zero-padded to 5 digits, so match that here
safetensors_path = out_dir / f"gaussians_{iteration:05d}.safetensors"
ply_to_safetensors(candidates[iteration], safetensors_path)
print(f"  gaussians   -> {safetensors_path}")

# * cfg_args -> config.json, falling back to GRay defaults where 3DGS is silent
cfg_path = model_dir / "cfg_args"
cfg = eval(cfg_path.read_text()) if cfg_path.exists() else Namespace()
source_path = getattr(cfg, "source_path", "")
resolution = getattr(cfg, "resolution", 1)
if not resolution or resolution < 1:
    resolution = 1
sh_degree = getattr(cfg, "sh_degree", 3)
config = asdict(Config(source_path=source_path, model_path=getattr(cfg, "model_path", "")))
config.update(
    {
        "downsampling": resolution,
        "images_dir": f"images_{resolution}",
        "point_cloud_file": getattr(cfg, "pc", "point_cloud.safetensors"),
        "eval": bool(getattr(cfg, "eval", True)),
        "bg_color": [1.0, 1.0, 1.0] if getattr(cfg, "white_background", False) else [0.0, 0.0, 0.0],
        "sh": sh_degree > 0,
        "sh_max_degree": sh_degree,
    }
)
(out_dir / "config.json").write_text(json.dumps(config, indent=4))
print(f"  config.json -> {out_dir / 'config.json'}")

# * cameras.json: 3DGS {id,rotation,position,fx,fy} -> GRay {uid,R,T,origin,fov_*}
cameras_path = model_dir / "cameras.json"
if cameras_path.exists():
    images_dir = config["images_dir"]
    gray_cameras = []
    for cam in json.loads(cameras_path.read_text()):
        rotation = np.asarray(cam["rotation"], dtype=np.float64)  # * camera-to-world
        position = np.asarray(cam["position"], dtype=np.float64)  # * camera center in world
        width, height = int(cam["width"]), int(cam["height"])
        img_name = cam.get("img_name", "")
        gray_cameras.append(
            {
                "uid": int(cam.get("id", 0)),
                "R": rotation.tolist(),
                # * GRay stores origin = -R @ T, so T = -R^T @ origin
                "T": (-rotation.T @ position).tolist(),
                "origin": position.tolist(),
                "fov_y": 2.0 * math.atan(height / (2.0 * float(cam["fy"]))),
                "fov_x": 2.0 * math.atan(width / (2.0 * float(cam["fx"]))),
                "image_path": os.path.join(source_path, images_dir, img_name) if source_path else img_name,
                "image_name": img_name,
                "image_width": width,
                "image_height": height,
                "is_test": False,
            }
        )
    (out_dir / "cameras.json").write_text(json.dumps(gray_cameras, indent=4))
    print(f"  cameras.json-> {out_dir / 'cameras.json'}  ({len(gray_cameras)} cameras)")
else:
    print(f"  WARNING: no cameras.json in {model_dir}")

print(f"Converted 3DGS model -> GRay scene at {out_dir}")
