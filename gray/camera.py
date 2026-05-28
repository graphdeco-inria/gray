from __future__ import annotations

# * Important: minimal imports so remote viewer can import this file without installing all dependencies
from dataclasses import dataclass
import os
import numpy as np


@dataclass
class CameraInfo:
    uid: int
    R: np.ndarray
    T: np.ndarray
    origin: np.ndarray
    fov_y: np.ndarray
    fov_x: np.ndarray
    image_path: str
    image_name: str
    image_width: int
    image_height: int
    is_test: bool

    @staticmethod
    def from_colmap(cfg, key, extr, intr, is_test: bool):
        from gray.utils import focal2fov
        from PIL import Image
        import gray.colmap as colmap

        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(colmap.qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        origin = -R @ T
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            fov_y = focal2fov(focal_length_x, height)
            fov_x = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            fov_y = focal2fov(focal_length_y, height)
            fov_x = focal2fov(focal_length_x, width)
        else:
            assert False, (
                "Colmap camera model not handled: only undistorted camera (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            )

        if os.path.isabs(extr.name):
            image_path = extr.name
        else:
            images_dir_name = "images" if cfg.images_dir is None else cfg.images_dir
            image_path = os.path.join(cfg.source_path, images_dir_name, extr.name)
        image_name = extr.name

        base, ext = np.path.splitext(image_path)
        if ext.lower() != ".png":
            image_path = base + ".png"
            image_name = np.path.splitext(image_name)[0] + ".png"

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        return CameraInfo(
            uid=uid,
            R=R,
            T=T,
            origin=origin,
            fov_y=fov_y,
            fov_x=fov_x,
            image_path=image_path,
            image_name=image_name,
            image_width=image_width,
            image_height=image_height,
            is_test=is_test,
        )

    @staticmethod
    def from_json(json_data: dict):
        kwargs = {}
        for field in CameraInfo.__dataclass_fields__:
            value = json_data.get(field)
            if isinstance(value, list):
                kwargs[field] = np.array(value)
            else:
                kwargs[field] = value
        return CameraInfo(**kwargs)

    def to_json(self):
        result = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, np.ndarray):
                result[field] = value.tolist()
            else:
                result[field] = value
        return result

    def origin_cuda(self):
        """Returns the camera origin cached as a CUDA tensor"""
        import torch

        tensor = getattr(self, "_origin_cuda", None)
        if tensor is None:
            tensor = torch.from_numpy(np.asarray(self.origin, dtype=np.float32)).cuda()
            self._origin_cuda = tensor
        return tensor

    def rotation_c2w_blender_cuda(self):
        """Returns the rotation from camera to world coordinates cached as a CUDA tensor, in Blender's coordinate system"""
        import torch

        tensor = getattr(self, "_rotation_c2w_blender_cuda", None)
        if tensor is None:
            rotation = -np.asarray(self.R, dtype=np.float32).copy()
            rotation[:, 0] *= -1
            tensor = torch.from_numpy(rotation).cuda()
            self._rotation_c2w_blender_cuda = tensor
        return tensor