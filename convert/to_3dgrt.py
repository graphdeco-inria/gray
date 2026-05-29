"""
Convert a GRay scene into a 3D Gaussian Ray Tracing (3DGRT / nv-tlabs 3dgrut) checkpoint.

GRay and 3dgrut store gaussians with the same conventions (log-scale, logit density/opacity,
wxyz quaternions, and (rgb-0.5)/C0 spherical harmonics), so this writes a 3dgrut checkpoint
(.pt) directly from GRay's safetensors -- no .ply intermediate and without importing the 3dgrut
package:

    <out>/training/ckpt_<iter>.pt    3dgrut checkpoint (render.py loads this)
    <out>/meta.json                  {source_path, downsample_factor, sh_degree, iteration}

A 3dgrut checkpoint also needs a valid OmegaConf config (render settings, optimizer spec,
background, scene_extent, ...). Rather than recompose all of that, we lift it from any existing
3dgrut checkpoint passed as --template-checkpoint, overriding only the dataset path, downsample
factor and SH degree; the gaussian tensors and a fresh optimizer state come from the GRay scene.
Render it inside the 3dgrut docker image with:

    python render.py --checkpoint <out>/training/ckpt_<iter>.pt

The tensor/optimizer layout below is reproduced from nv-tlabs/3dgrut (threedgrut/model/model.py
init_from_pretrained_point_cloud + setup_optimizer):
    SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0
"""

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import re
from typing import Optional

import safetensors.torch
import torch
import tyro
from omegaconf import OmegaConf
from tyro.conf import Positional

# * SH band-0 normalisation constant (used only for SH-less GRay models)
C0 = 0.28209479177387814
# * default template: any trained 3dgrut checkpoint works (used only for its config/background)
DEFAULT_TEMPLATE = "~/Desktop/3dgrut/runs_sfm/bicycle/training/ours_7000/ckpt_7000.pt"
# * 3dgrut generalized-gaussian degrees with a closed form (threedgrut gaussianParticles.cuh)
SUPPORTED_3DGRT_DEGREES = (1, 2, 3, 4, 5, 8)
# * 3dgrut gaussian attribute -> conf.model.optimize_* flag (threedgrut/model/model.py)
ATTR_OPTIMIZE_FLAG = {
    "positions": "optimize_position",
    "density": "optimize_density",
    "features_albedo": "optimize_features_albedo",
    "features_specular": "optimize_features_specular",
    "rotation": "optimize_rotation",
    "scale": "optimize_scale",
}


@dataclass
class ConversionCLI:
    # * GRay scene directory (config.json, gaussians_*.safetensors)
    input_path: Positional[str]
    # * 3DGRT bundle directory to create (default: <input>/3dgrt)
    output_path: Positional[Optional[str]] = None
    # * Which gaussians_<iter>.safetensors to convert (default: highest iteration)
    iteration: Optional[int] = None
    # * COLMAP dataset path as seen by 3dgrut when rendering (default: config.json source_path)
    source_path: Optional[str] = None
    # * Downsample factor 3dgrut renders at; keep it high to stay GPU-light (default: 8)
    downsample: int = 8
    # * Any existing 3dgrut checkpoint, used only for its config / background / scene_extent
    template_checkpoint: str = DEFAULT_TEMPLATE
    # * Rescale gaussians so 3dgrut's generalized-gaussian kernel EXACTLY matches GRay's exp_power
    # * kernel (removes 3dgrt's extra blur). Disable for an exact, round-trip-lossless transfer.
    match_kernel: bool = True


cli = tyro.cli(ConversionCLI)
scene_dir = Path(cli.input_path)
if not scene_dir.is_dir():
    raise NotADirectoryError(f"Input must be a GRay scene directory: {scene_dir}")

out_dir = Path(cli.output_path) if cli.output_path else scene_dir / "3dgrt"
(out_dir / "training").mkdir(parents=True, exist_ok=True)

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

gaussians = safetensors.torch.load_file(str(candidates[iteration]))
missing = sorted({"mean", "rotation", "scale", "opacity"} - gaussians.keys())
if missing:
    raise ValueError(f"safetensors missing required gaussian tensors: {', '.join(missing)}")
num = gaussians["mean"].shape[0]

# * GRay -> 3dgrut gaussians (same parameter space, values copy directly). GRay's sh_coeffs_rest
# * is (N, K, 3) coeff-major, which flattens exactly to 3dgrut's (N, K*3) features_specular.
if "sh_coeffs_dc" in gaussians and "sh_coeffs_rest" in gaussians:
    albedo = gaussians["sh_coeffs_dc"].reshape(num, 3)
    num_coeffs = gaussians["sh_coeffs_rest"].shape[1]
    sh_degree = math.isqrt(num_coeffs + 1) - 1
    specular = gaussians["sh_coeffs_rest"].reshape(num, num_coeffs * 3)
elif "channels" in gaussians:
    # * No SH bands: fold the first 3 (RGB) channels into the SH DC term
    albedo = (gaussians["channels"][:, :3] - 0.5) / C0
    specular = torch.zeros((num, 0), dtype=torch.float32)
    sh_degree = 0
else:
    raise ValueError("safetensors missing both sh_coeffs_* and channels")

# * GRay renders rho = exp(-M^(2p)/(2p)) (p = exp_power) while 3dgrut degree n renders
# * rho = exp(-(4.5/3^n) M^n). They are the same kernel family iff n = 2p, and then 3dgrut's
# * wider (blurrier) normalisation is removed EXACTLY by multiplying every scale by
# * f = (4.5 n / 3^n)^(1/n) (e.g. p=2 -> n=4 -> f=(2/9)^(1/4) ~= 0.687). Scales are log-space.
config_path = scene_dir / "config.json"
gray_config = json.loads(config_path.read_text()) if config_path.exists() else {}
exp_power = float(gray_config.get("exp_power", 2.0))
kernel_degree = round(2 * exp_power)
# * An exact match needs n = 2p to be one of 3dgrut's closed-form degrees. When it is, we both set
# * the render degree and apply the scale offset (which is exactly 0 for p=1, where the kernels
# * already coincide -- but the degree must still be switched from the template's default).
matched = (
    cli.match_kernel
    and abs(kernel_degree - 2 * exp_power) <= 1e-6
    and kernel_degree in SUPPORTED_3DGRT_DEGREES
)
if cli.match_kernel and not matched:
    raise ValueError(
        f"GRay exp_power={exp_power} has no exact 3dgrut kernel (would need degree {2 * exp_power}; "
        f"3dgrut supports degrees {SUPPORTED_3DGRT_DEGREES}). Re-run with --no-match-kernel to "
        f"convert without kernel matching (3dgrut will then render blurrier than GRay)."
    )
log_scale_offset = (
    math.log((4.5 * kernel_degree / 3 ** kernel_degree) ** (1.0 / kernel_degree)) if matched else 0.0
)
if matched:
    # * Sanity: rescaling scales by f = exp(offset) must turn 3dgrut's degree-n coefficient
    # * (-4.5/3^n) into GRay's (-1/(2p)), i.e. the two kernels become identical.
    f = math.exp(log_scale_offset)
    assert math.isclose(4.5 / 3 ** kernel_degree / f**kernel_degree, 1.0 / (2 * exp_power), rel_tol=1e-9), (
        f"kernel-match factor {f} does not reproduce GRay's exp_power={exp_power} kernel"
    )

scale_t = gaussians["scale"].float().contiguous()
if matched:
    scale_t = scale_t + log_scale_offset

# * 3dgrut's render.py loads the checkpoint without remapping devices and feeds the tensors
# * straight to its CUDA/OptiX tracer, so they must be saved on the GPU.
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required: 3dgrut loads checkpoint tensors to their saved device without "
        "remapping, so the gaussians must be saved on the GPU. Run this on a CUDA machine."
    )
model_params = {
    "positions": torch.nn.Parameter(gaussians["mean"].float().contiguous().cuda()),
    "density": torch.nn.Parameter(gaussians["opacity"].float().contiguous().cuda()),
    "scale": torch.nn.Parameter(scale_t.cuda()),
    "rotation": torch.nn.Parameter(gaussians["rotation"].float().contiguous().cuda()),
    "features_albedo": torch.nn.Parameter(albedo.float().contiguous().cuda()),
    "features_specular": torch.nn.Parameter(specular.float().contiguous().cuda()),
}

# * Lift config + non-gaussian state from a template checkpoint (mmap: only the small config /
# * background / scalars are read, not the template's gaussian or optimizer tensors).
template = torch.load(os.path.expanduser(cli.template_checkpoint), map_location="cpu",
                      weights_only=False, mmap=True)
conf = template["config"]
OmegaConf.set_struct(conf, False)
source_path = cli.source_path if cli.source_path is not None else gray_config.get("source_path", "")
conf.path = source_path
conf.dataset.downsample_factor = cli.downsample
conf.model.progressive_training.max_n_features = sh_degree
if matched:
    conf.render.particle_kernel_degree = kernel_degree  # * GRay kernel family is n = 2*exp_power

# * Fresh Adam optimizer state matching the new gaussians, in conf.optimizer.params order and
# * gated by conf.model.optimize_* (mirrors threedgrut setup_optimizer). render.py rebuilds the
# * optimizer from conf and load_state_dict()s this, so an unstepped/empty state is enough.
groups = [
    {"params": [model_params[name]], "name": name}
    for name in conf.optimizer.params
    if name in model_params and bool(conf.model.get(ATTR_OPTIMIZE_FLAG[name], True))
]
optimizer = torch.optim.Adam(groups, lr=0.0, eps=float(conf.optimizer.eps))

checkpoint = {
    **model_params,
    "n_active_features": sh_degree,
    "max_n_features": sh_degree,
    "progressive_training": bool(template.get("progressive_training", False)),
    "scene_extent": template["scene_extent"],
    "background": template["background"],
    "optimizer": optimizer.state_dict(),
    "config": conf,
    "global_step": iteration,
    "epoch": 0,
}
for key in ("feature_dim_increase_interval", "feature_dim_increase_step"):
    if key in template:
        checkpoint[key] = template[key]

ckpt_path = out_dir / "training" / f"ckpt_{iteration}.pt"
torch.save(checkpoint, ckpt_path)

meta = {
    "source_path": source_path,
    "downsample_factor": cli.downsample,
    "sh_degree": sh_degree,
    "iteration": iteration,
    "kernel_matched": matched,
    "kernel_degree": kernel_degree if matched else None,
    "scale_multiplier": math.exp(log_scale_offset),
}
(out_dir / "meta.json").write_text(json.dumps(meta, indent=4))
print(f"  checkpoint  -> {ckpt_path}  ({num} gaussians, sh_degree={sh_degree})")
print(f"  meta.json   -> {out_dir / 'meta.json'}  {meta}")
print(f"Converted GRay scene -> 3DGRT checkpoint at {out_dir}")
print("Render it inside the 3dgrut docker image (downsample is baked into the checkpoint config):")
print(f"    python render.py --checkpoint {ckpt_path}")
