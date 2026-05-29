"""
Convert a 3D Gaussian Ray Tracing (3DGRT / nv-tlabs 3dgrut) checkpoint into a GRay scene.

Reverse of to_3dgrt.py. Reads a 3dgrut checkpoint (.pt) directly -- no .ply intermediate and
without importing the 3dgrut package -- and writes a GRay scene:

    <out>/gaussians_<iter>.safetensors
    <out>/config.json

GRay and 3dgrut share the gaussian parameter conventions, so the tensors copy directly. The
checkpoint embeds an OmegaConf config (hence the omegaconf dependency, used here only to read the
dataset path / downsample factor for config.json).

The tensor layout below is reproduced from nv-tlabs/3dgrut (threedgrut/model/model.py,
threedgrut/export/ply_exporter.py):
    SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import sys
from typing import Optional

import safetensors.torch
import torch
import tyro
from tyro.conf import Positional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gray.config import Config


@dataclass
class ConversionCLI:
    # * 3dgrut checkpoint .pt (or a directory containing training/ckpt_*.pt)
    input_path: Positional[str]
    # * GRay scene directory to create (default: <checkpoint dir>/gray)
    output_path: Positional[Optional[str]] = None
    # * Iteration label for gaussians_<iter>.safetensors (default: checkpoint global_step)
    iteration: Optional[int] = None


cli = tyro.cli(ConversionCLI)
in_path = Path(cli.input_path)
if in_path.is_file():
    ckpt_path = in_path
else:
    found = sorted(in_path.glob("**/ckpt_*.pt"))
    if not found:
        raise FileNotFoundError(f"No ckpt_*.pt found under {in_path}")
    ckpt_path = found[-1]

# * mmap so only the gaussian tensors are read (not the optimizer state); weights_only=False is
# * required because the checkpoint embeds an OmegaConf config object.
ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False, mmap=True)
num = ck["positions"].shape[0]
max_n = int(ck["max_n_features"])
num_coeffs = (max_n + 1) ** 2 - 1

# * 3dgrut features_specular is (N, K*3) coeff-major -> GRay sh_coeffs_rest (N, K, 3)
tensors = {
    "mean": ck["positions"].detach().float().contiguous(),
    "opacity": ck["density"].detach().float().contiguous(),
    "rotation": ck["rotation"].detach().float().contiguous(),
    "scale": ck["scale"].detach().float().contiguous(),
    "sh_coeffs_dc": ck["features_albedo"].detach().float().reshape(num, 1, 3).contiguous(),
    "sh_coeffs_rest": ck["features_specular"].detach().float().reshape(num, num_coeffs, 3).contiguous(),
    "current_sh_degree": torch.tensor([max_n], dtype=torch.int32),
}

iteration = cli.iteration if cli.iteration is not None else int(ck.get("global_step", 30000))
out_dir = Path(cli.output_path) if cli.output_path else ckpt_path.parent / "gray"
out_dir.mkdir(parents=True, exist_ok=True)
# * GRay loads gaussians_<iter> zero-padded to 5 digits, so match that here
safetensors_path = out_dir / f"gaussians_{iteration:05d}.safetensors"
safetensors.torch.save_file(tensors, str(safetensors_path))
print(f"  gaussians   -> {safetensors_path}  ({num} gaussians, sh_degree={max_n})")

# * Recover dataset info from the checkpoint's config for config.json (best effort)
source_path = ""
resolution = 1
conf = ck.get("config")
if conf is not None:
    try:
        source_path = str(conf.path)
        resolution = int(conf.dataset.downsample_factor) or 1
    except Exception:
        pass
config = asdict(Config(source_path=source_path, model_path=str(out_dir)))
config.update(
    {
        "downsampling": resolution,
        "images_dir": f"images_{resolution}",
        "eval": True,
        "bg_color": [0.0, 0.0, 0.0],
        "sh": max_n > 0,
        "sh_max_degree": max_n,
    }
)
(out_dir / "config.json").write_text(json.dumps(config, indent=4))
print(f"  config.json -> {out_dir / 'config.json'}")
print(f"Converted 3DGRT checkpoint -> GRay scene at {out_dir}")
