from dataclasses import dataclass, field
import tyro
from tyro.conf import arg
from typing import Annotated, List, Optional, Literal


@dataclass
class DatasetConfig:
    source_path: Annotated[str, arg(aliases=["-s"])]  
    model_path: Annotated[str, arg(aliases=["-m"])]  

    downsampling: Annotated[str, arg(aliases=["-r"])] = 1  # * Integer downsampling factor
    images_dir: Annotated[str, arg(aliases=["-i"])] = "images_{downsampling}" # * Relative to source_path or absolute
    
    point_cloud_file: Annotated[str, arg(aliases=["-p"])] = "point_cloud.safetensors" # * Relative to source_path or absolute
    
    eval: bool = False  

    def __post_init__(self):
        # * Allow using other settings when specifying paths e.g. {downsampling} in images_dir
        self.images_dir = self.images_dir.format(
            downsampling=self.downsampling, source_path=self.source_path
        )
        self.point_cloud_file = self.point_cloud_file.format(
            downsampling=self.downsampling, source_path=self.source_path, images_dir=self.images_dir
        )


@dataclass
class RaytracerConfig:
    # * Render settings
    render_depth: bool = False  
    preview_train_image_name: Optional[str] = None 
    preview_test_image_name: Optional[str] = None

    # * Logging
    preview_iters: List[int] = field(default_factory=lambda: [1, 1_000, 2_500, 7_500, 15_000])
    test_iters: List[int] = field(default_factory=lambda: [7_500, 15_000])
    save_iters: List[int] = field(default_factory=lambda: [7_500, 15_000])
    log_loss_interval: int = 1000
    log_stats_interval: int = 1000
    viewer: bool = False  # * Open the viewer during training
    yes: Annotated[bool, arg(aliases=["-y"])] = # * Allows overwriting existing directories without prompt

    # * Memory use 
    ppll_forward_size: int = 300_000_000
    ppll_backward_size: int = 120_000_000

    # * Background color
    bg_color: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # * Raytracing quality
    exp_power: float = 2.0
    alpha_threshold: float = 0.01
    t_threshold: float = 0.03

    # * Init
    init_scale: float = 0.0005  # * Higher values may work better at low resolution
    init_opacity: float = 0.1
    init_binning: bool = True
    init_bin_size: float =  0.0015  # * Post-bugfix default; tuned to roughly match old behavior at 0.04

    # * Loss
    lambda_ssim: float = 0.2

    # * Optimization
    iterations: int = 15_000
    lr_mean_init: float = 0.00016
    lr_mean_final: float = 0.0000016
    lr_channels: float = 0.0025  # * Only used when SH are disabled
    lr_opacity_init: float = 0.025
    lr_opacity_final: float = 0.025
    lr_scale_init: float = 0.02
    lr_scale_final: float = 0.005
    lr_rotation_init: float = 0.004
    lr_rotation_final: float = 0.001
    lr_sh_dc_init: float = 0.04
    lr_sh_dc_final: float = 0.0025
    lr_sh_rest: float = 0.000625
    tiling: int = 1  # * Legacy setting retained for backwards compatibility
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-15
    sh_update_laziness: int = 1 # * Only step the SH non-dc coefficients every `sh_update_laziness` iterations
    lr_schedule_delay_mult: float = 0.01

    # * Pruning
    pruning: bool = True
    pruning_interval: int = 500  # * Was 100 in 3DGS
    pruning_from_iter: int = 500
    pruning_min_weight: float = 1e-7

    # * Performance settings
    rebuild_interval: int = 500

    # * Scale decay
    scale_decay: float = 0.999875

    # * SH settings
    sh: bool = True
    sh_init_degree: int = 0
    sh_max_degree: int = 3
    sh_increment_interval: int = 1000

    # * Exposure compensation
    exposure_comp_enabled: bool = False
    exposure_comp_lr_init: float = 0.001
    exposure_comp_lr_final: float = 0.0001
    exposure_comp_lr_delay_mult: float = 0.001
    exposure_comp_lr_max_steps: int = 5000

    # * MLP settings
    pre_mlp: bool = False
    pre_mlp_feature_size: int = 8
    pre_mlp_width: int = 128
    pre_mlp_layers: int = 4
    pre_mlp_lr: float = 1e-3
    pre_mlp_feature_lr: float = 3e-2
    pre_mlp_freq_bands: Optional[int] = 4
    post_mlp: bool = False
    post_mlp_width: int = 128  # * Wider can improve PSNR a bit
    post_mlp_layers: int = 5  # * Deeper may improve PSNR a bit
    post_mlp_lr: float = 1e-3
    post_mlp_freq_bands: Optional[int] = 1  # * Optimal value may be scene-dependent
    tcnn: bool = False

    def __post_init__(self):
        # * Ensure save_iters includes the final iteration
        if self.iterations not in self.save_iters:
            self.save_iters.append(self.iterations)
        if self.iterations not in self.test_iters:
            self.test_iters.append(self.iterations)
        if self.iterations not in self.preview_iters:
            self.preview_iters.append(self.iterations)

        # * Enforce valid configurations
        assert self.sh_init_degree <= self.sh_max_degree
        assert 0 <= self.sh_max_degree <= 3
        if self.sh:
            assert not self.pre_mlp, (
                "Spherical harmonics cannot be used with pre-MLP (choose either one)"
            )
            assert not self.post_mlp, "Spherical harmonics cannot be used with post-MLP"
        if not self.sh:
            self.sh_max_degree = 0


@dataclass
class Config(RaytracerConfig, DatasetConfig):
    def __post_init__(self):
        DatasetConfig.__post_init__(self)
        RaytracerConfig.__post_init__(self)
