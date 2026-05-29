from dataclasses import dataclass
from pathlib import Path
import math
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
from plyfile import PlyData, PlyElement
import safetensors.torch
import torch
import tyro
from tyro.conf import Positional


@dataclass
class ConversionCLI:
    input_path: Positional[str]
    output_path: Positional[Optional[str]] = None


def ply_to_safetensors(input_path: Path, output_path: Path) -> None:
    vertex = PlyData.read(input_path)["vertex"].data
    num_points = len(vertex)
    tensors = {
        "mean": torch.from_numpy(
            np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        ),
        "opacity": torch.from_numpy(np.expand_dims(vertex["opacity"], axis=1).astype(np.float32)),
        "rotation": torch.from_numpy(
            np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1).astype(
                np.float32
            )
        ),
        "scale": torch.from_numpy(
            np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(
                np.float32
            )
        ),
    }
    f_rest_keys = sorted(
        (name for name in vertex.dtype.fields if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )

    if f_rest_keys:
        if len(f_rest_keys) % 3 != 0:
            raise ValueError("PLY file has an invalid number of SH rest fields")

        sh_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(
            np.float32
        )[:, None, :]
        sh_rest = np.ascontiguousarray(
            np.stack([vertex[key] for key in f_rest_keys], axis=1)
            .astype(np.float32)
            .reshape(num_points, 3, -1)
            .transpose(0, 2, 1)
        )
        base = sh_rest.shape[1] + 1
        root = math.isqrt(base)
        if root * root != base:
            raise ValueError(f"Invalid SH rest term count: {sh_rest.shape[1]}")

        tensors["sh_coeffs_dc"] = torch.from_numpy(sh_dc)
        tensors["sh_coeffs_rest"] = torch.from_numpy(sh_rest)
        tensors["current_sh_degree"] = torch.tensor([root - 1], dtype=torch.int32)
    else:
        tensors["channels"] = torch.from_numpy(
            np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(
                np.float32
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(tensors, str(output_path))


def safetensors_to_ply(input_path: Path, output_path: Path) -> None:
    tensors = safetensors.torch.load_file(str(input_path))
    missing_keys = sorted({"mean", "rotation", "scale", "opacity"} - tensors.keys())
    if missing_keys:
        raise ValueError(
            "Input safetensors file is missing gaussian tensors: " + ", ".join(missing_keys)
        )

    points = tensors["mean"].detach().cpu().numpy().astype(np.float32)
    opacities = tensors["opacity"].detach().cpu().numpy().astype(np.float32)
    rotations = tensors["rotation"].detach().cpu().numpy().astype(np.float32)
    scales = tensors["scale"].detach().cpu().numpy().astype(np.float32)
    has_sh = "sh_coeffs_dc" in tensors and "sh_coeffs_rest" in tensors
    if has_sh:
        sh_dc = tensors["sh_coeffs_dc"].detach().cpu().numpy().astype(np.float32)
        sh_rest = tensors["sh_coeffs_rest"].detach().cpu().numpy().astype(np.float32)
    elif "channels" in tensors:
        channels = tensors["channels"].detach().cpu().numpy().astype(np.float32)
    else:
        raise ValueError("Input safetensors file is missing either channels or SH coefficients")

    dtype_fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]
    if has_sh:
        for index in range(sh_rest.shape[1] * sh_rest.shape[2]):
            dtype_fields.append((f"f_rest_{index}", "f4"))

    vertex = np.empty(points.shape[0], dtype=dtype_fields)
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["opacity"] = opacities[:, 0]
    vertex["rot_0"] = rotations[:, 0]
    vertex["rot_1"] = rotations[:, 1]
    vertex["rot_2"] = rotations[:, 2]
    vertex["rot_3"] = rotations[:, 3]
    vertex["scale_0"] = scales[:, 0]
    vertex["scale_1"] = scales[:, 1]
    vertex["scale_2"] = scales[:, 2]

    if has_sh:
        vertex["f_dc_0"] = sh_dc[:, 0, 0]
        vertex["f_dc_1"] = sh_dc[:, 0, 1]
        vertex["f_dc_2"] = sh_dc[:, 0, 2]
        sh_rest_flat = np.ascontiguousarray(
            sh_rest.transpose(0, 2, 1).reshape(sh_rest.shape[0], -1)
        )
        for index in range(sh_rest_flat.shape[1]):
            vertex[f"f_rest_{index}"] = sh_rest_flat[:, index]
    else:
        vertex["f_dc_0"] = channels[:, 0]
        vertex["f_dc_1"] = channels[:, 1]
        vertex["f_dc_2"] = channels[:, 2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(output_path))


if __name__ == "__main__":
    cli = tyro.cli(ConversionCLI)
    input_path = Path(cli.input_path)
    if input_path.suffix not in {".ply", ".safetensors"}:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")

    if cli.output_path is not None:
        output_path = Path(cli.output_path)
    elif input_path.suffix == ".ply":
        output_path = input_path.with_suffix(".safetensors")
    else:
        output_path = input_path.with_suffix(".ply")

    if input_path.suffix == output_path.suffix:
        raise ValueError("Output path must use the opposite file format")

    if input_path.suffix == ".ply":
        ply_to_safetensors(input_path, output_path)

        with TemporaryDirectory(dir=output_path.parent) as tmp_dir:
            roundtrip_path = Path(tmp_dir) / "cycle_check.ply"
            safetensors_to_ply(output_path, roundtrip_path)

            original = PlyData.read(input_path)["vertex"].data
            roundtrip = PlyData.read(roundtrip_path)["vertex"].data
            if original.dtype.names != roundtrip.dtype.names:
                raise ValueError("Cycle consistency failed: PLY fields changed after round-trip")
            for name in original.dtype.names:
                if not np.array_equal(original[name], roundtrip[name]):
                    raise ValueError(
                        f"Cycle consistency failed: PLY field '{name}' changed after round-trip"
                    )
    else:
        safetensors_to_ply(input_path, output_path)

        with TemporaryDirectory(dir=output_path.parent) as tmp_dir:
            roundtrip_path = Path(tmp_dir) / "cycle_check.safetensors"
            ply_to_safetensors(output_path, roundtrip_path)

            original = safetensors.torch.load_file(str(input_path))
            roundtrip = safetensors.torch.load_file(str(roundtrip_path))
            for tensors in (original, roundtrip):
                tensors.pop("bg_color", None)
                tensors.pop("image_width", None)
                tensors.pop("image_height", None)
                has_sh = (
                    "sh_coeffs_rest" in tensors
                    and tensors["sh_coeffs_rest"].ndim == 3
                    and tensors["sh_coeffs_rest"].shape[1] > 0
                )
                if has_sh:
                    tensors.pop("channels", None)
                    tensors.pop("current_sh_degree", None)
                else:
                    tensors.pop("sh_coeffs_dc", None)
                    tensors.pop("sh_coeffs_rest", None)
                    tensors.pop("current_sh_degree", None)

            if original.keys() != roundtrip.keys():
                raise ValueError(
                    "Cycle consistency failed: safetensors keys changed after round-trip"
                )
            for name in sorted(original):
                if original[name].shape != roundtrip[name].shape or not torch.equal(
                    original[name], roundtrip[name]
                ):
                    raise ValueError(
                        f"Cycle consistency failed: safetensors tensor '{name}' changed after round-trip"
                    )

    print(f"Saved {output_path}")