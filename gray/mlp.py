from gray.config import Config
from gray.camera import CameraInfo

import torch
import torch.nn as nn


class PreMLP(nn.Module):
    def __init__(self, cfg: Config, gaussians):
        super().__init__()
        self.cfg = cfg
        self.gaussians = gaussians
        assert cfg.pre_mlp_layers >= 2, "MLP must have at least 2 layers"
        self.gaussian_features = nn.Parameter(
            torch.randn(
                gaussians.mean.shape[0],
                cfg.pre_mlp_feature_size,
                dtype=torch.float32,
                device="cuda",
            ).to(torch.bfloat16)
        )

        if cfg.pre_mlp_freq_bands is None:
            input_size = cfg.pre_mlp_feature_size
        else:
            input_size = cfg.pre_mlp_feature_size + positional_encoding_size(
                3, self.cfg.pre_mlp_freq_bands
            )

        output_size = gaussians.channels.shape[1]

        if self.cfg.tcnn:
            import tinycudann as tcnn

            self.network = tcnn.NetworkWithInputEncoding(
                input_size,
                3,
                {"otype": "Identity"},
                {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": cfg.pre_mlp_width,
                    "n_hidden_layers": cfg.pre_mlp_layers - 2,
                },
            ).float()
        else:
            self.network = (
                nn.Sequential(
                    nn.Linear(input_size, cfg.pre_mlp_width),
                    nn.ReLU(),
                    *[
                        nn.Sequential(
                            nn.Linear(cfg.pre_mlp_width, cfg.pre_mlp_width),
                            nn.ReLU(inplace=True),
                        )
                        for _ in range(cfg.pre_mlp_layers - 2)
                    ],
                    nn.Linear(cfg.pre_mlp_width, output_size),
                )
                .cuda()
                .to(torch.bfloat16)
            )

            # * Init weights
            for m in self.network:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # * Setup optimizer
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.gaussian_features,
                    "lr": cfg.pre_mlp_feature_lr,
                },
                {
                    "params": self.network.parameters(),
                    "lr": cfg.pre_mlp_lr,
                },
            ]
        )

        # * Keep track of the last output for backprop
        self.output = None

    def initialize(self):
        self.register_buffer("init_channels", self.gaussians.channels.clone().detach())

    def forward(self, cam_info: CameraInfo):
        x = self.gaussian_features.to(torch.bfloat16)

        if self.cfg.pre_mlp_freq_bands is not None:
            view_directions = nn.functional.normalize(
                self.gaussians.mean
                - torch.from_numpy(cam_info.origin).to(torch.bfloat16).unsqueeze(0).cuda(),
                dim=-1,
            ).to(torch.bfloat16)
            encoding = positional_encoding(view_directions, self.cfg.pre_mlp_freq_bands)
            x = torch.cat([encoding, self.gaussian_features], dim=1)

        self.output = self.network(x)

        with torch.no_grad():
            self.gaussians.channels.copy_(self.init_channels + self.output.detach())

    @torch.no_grad()
    def prune(self, mask):
        self.gaussian_features.data = self.gaussian_features[mask]
        self.init_channels = self.init_channels[mask]
        state = self.optimizer.state[self.gaussian_features]
        state["exp_avg"] = state["exp_avg"][mask]
        state["exp_avg_sq"] = state["exp_avg_sq"][mask]

    def densify(self, mask):
        self.gaussian_features.data = torch.cat(
            [self.gaussian_features, self.gaussian_features[mask]], dim=0
        )
        self.init_channels = torch.cat([self.init_channels, self.init_channels[mask]], dim=0)
        state = self.optimizer.state[self.gaussian_features]
        state["exp_avg"] = torch.cat([state["exp_avg"], state["exp_avg"][mask]], dim=0)
        state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], state["exp_avg_sq"][mask]], dim=0)

    @torch.no_grad()
    def sort(self, permutation: torch.Tensor):
        self.gaussian_features.data = self.gaussian_features[permutation]
        self.init_channels = self.init_channels[permutation]
        state = self.optimizer.state[self.gaussian_features]
        state["exp_avg"] = state["exp_avg"][permutation]
        state["exp_avg_sq"] = state["exp_avg_sq"][permutation]

    def step(self):
        grads = self.gaussians.channels.grad
        self.output.backward(grads)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.output = None
        grads.zero_()


class PostMLP(nn.Module):
    def __init__(self, cfg: Config, num_channels: int):
        super().__init__()
        self.cfg = cfg
        assert cfg.post_mlp_layers >= 2, "MLP must have at least 2 layers"

        if cfg.post_mlp_freq_bands is None:
            input_size = num_channels
        else:
            input_size = num_channels + positional_encoding_size(6, self.cfg.post_mlp_freq_bands)

        if self.cfg.tcnn:
            import tinycudann as tcnn

            self.network = tcnn.NetworkWithInputEncoding(
                input_size,
                3,
                {"otype": "Identity"},
                {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": cfg.post_mlp_width,
                    "n_hidden_layers": cfg.post_mlp_layers - 2,
                },
            )
        else:
            self.network = (
                nn.Sequential(
                    nn.Conv2d(input_size, cfg.post_mlp_width, kernel_size=1),
                    nn.ReLU(),
                    *[
                        nn.Sequential(
                            nn.Conv2d(cfg.post_mlp_width, cfg.post_mlp_width, kernel_size=1),
                            nn.ReLU(inplace=True),
                        )
                        for _ in range(cfg.post_mlp_layers - 2)
                    ],
                    nn.Conv2d(cfg.post_mlp_width, 3, kernel_size=1),
                )
                .cuda()
                .to(torch.bfloat16)
            )

            # * Init weights
            for m in self.network:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.post_mlp_lr)

    def forward(self, x, view_direction, hit_position):
        if self.cfg.post_mlp_freq_bands is None:
            x = x * 2 - 1
        else:
            encoding = positional_encoding(
                torch.cat([view_direction, hit_position], dim=0), self.cfg.post_mlp_freq_bands
            )
            x = torch.cat([x * 2 - 1, encoding], dim=0)

        if self.cfg.tcnn:
            out = self.network(x.flatten(1).moveaxis(0, -1))
            return out.reshape(x.shape[1], x.shape[2], 3).moveaxis(-1, 0)
        else:
            x = x.to(torch.bfloat16)
            return self.network(x).to(torch.float32).squeeze(-1)

    def step(self):
        self.optimizer.step()
        self.network.zero_grad()


def positional_encoding(image: torch.Tensor, k: int) -> torch.Tensor:
    """
    Sinusoidal positional encoding for a multi-channel 2D image or a batch of vectors,
    including the original image in the output.

    Args:
        image: Tensor of shape (C, H, W) or (N, C), float dtype recommended.
        k: Number of frequency bands. If k=0, returns the image unchanged.

    Returns:
        Tensor of shape (C + 2*C*k, H, W) if input is (C, H, W) and k>0,
        or (N, C + 2*C*k) if input is (N, C) and k>0,
        else same shape as input.
    """
    assert image.ndim in (2, 3), f"expected 2D (N, C) or 3D (C, H, W), got {tuple(image.shape)}"
    assert isinstance(k, int) and k >= 0, f"k must be a non-negative int, got {k}"

    if k == 0:
        return image

    if image.ndim == 3:
        # (C, H, W)
        freq_bands = 2 ** torch.arange(k, device=image.device, dtype=image.dtype)  # (F,)
        x = image.unsqueeze(-1) * freq_bands  # (C, H, W, F)
        x = x.permute(0, 3, 1, 2).reshape(
            image.shape[0] * k, image.shape[1], image.shape[2]
        )  # (C*F, H, W)
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=0)  # (2*C*F, H, W)
        return torch.cat([image, enc], dim=0)  # (C + 2*C*F, H, W)
    else:
        # (N, C)
        freq_bands = 2 ** torch.arange(k, device=image.device, dtype=image.dtype)  # (F,)
        x = image.unsqueeze(-1) * freq_bands  # (N, C, F)
        x = x.reshape(image.shape[0], image.shape[1] * k)  # (N, C*F)
        enc = torch.cat([torch.sin(x), torch.cos(x)], dim=1)  # (N, 2*C*F)
        return torch.cat([image, enc], dim=1)  # (N, C + 2*C*F)


def positional_encoding_size(c: int, k: int) -> int:
    """
    Returns the number of channels after positional encoding
    when the original image is also included.

    Args:
        c: Number of input channels.
        k: Number of frequency bands.

    Returns:
        Number of output channels after encoding.
    """
    if k == 0:
        return c
    return c + 2 * c * k
