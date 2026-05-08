from gray.config import Config
from gray.scene import SceneInfo, CameraInfo

import torch
import torch.nn as nn
import itertools


class ExposureComp(nn.Module):
    def __init__(self, cfg: Config, train_cameras: list[CameraInfo]):
        super().__init__()
        self.cfg = cfg

        self.shifts = {
            cam.image_name.replace(".", "_"): torch.zeros(1, device="cuda", requires_grad=True)
            for cam in train_cameras
        }
        self.scales = {
            cam.image_name.replace(".", "_"): torch.ones(1, device="cuda", requires_grad=True)
            for cam in train_cameras
        }

        self.optimizer = torch.optim.Adam(
            itertools.chain(self.scales.values(), self.shifts.values()),
            lr=cfg.exposure_comp_lr_init,
        )

    def __call__(self, image: torch.Tensor, image_name: str):
        name = image_name.replace(".", "_")
        return self.scales[name] * image + self.shifts[name]

    def set_lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
