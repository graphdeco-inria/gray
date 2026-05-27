from gray.imports import *
from gray.config import RaytracerConfig
from gray.scene import SceneInfo, CameraInfo, BasicPointCloud
from gray.mlp import PreMLP, PostMLP
from gray.exposure_comp import ExposureComp


class Raytracer(torch.nn.Module):
    __PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == "win32":
        # Windows: look for gray.dll in build\Release\
        __PROJECT_DIR = os.path.dirname(__PARENT_DIR)
        LIB_PATH = os.path.join(__PROJECT_DIR, "build", "Release", "gray.dll")
    elif os.path.exists(os.path.join(__PARENT_DIR, "libgray.so")):
        # * In a pip module, the shared library is dumped next to this file
        LIB_PATH = os.path.join(__PARENT_DIR, "libgray.so")
    else:
        # * During development, the shared library is in the project's build folder
        LIB_PATH = os.path.join(
            os.path.dirname(__PARENT_DIR),
            "build",
            "libgray.so",
        )
    LOADED = False

    def __init__(
        self,
        cfg: RaytracerConfig,
        num_points: int,
        image_width: int,
        image_height: int,
        inference_only: bool = False,
    ):
        "Note that you should call `from_ply` or `from_point_cloud` to initialize the raytracer with actual data."
        super().__init__()

        self.cfg = cfg
        self.image_width = image_width
        self.image_height = image_height

        # * Load the CUDA module
        torch.classes.load_library(Raytracer.LIB_PATH)
        Raytracer.LOADED = True

        self.cuda_module = torch.classes.gray.Raytracer(
            image_width,
            image_height,
            num_points,
            cfg.sh_max_degree,
            cfg.ppll_forward_size,
            cfg.ppll_backward_size,
            inference_only,
        )

        # * Decide background color
        self.bg_color = torch.tensor(cfg.bg_color)

        # * Setup config
        config = self.cuda_module.get_config()
        config.alpha_threshold.fill_(cfg.alpha_threshold)
        config.t_threshold.fill_(cfg.t_threshold)
        config.exp_power.fill_(cfg.exp_power)
        config.render_depth.fill_(cfg.render_depth)
        config.background_channels.copy_(self.bg_color)
        config.enable_sh.fill_(cfg.sh)
        config.needs_ray_output.fill_(cfg.post_mlp)

        # * Only optimize the channels if they aren't a viewpoint-depenent output color
        config.update_channels.fill_(not cfg.pre_mlp and not cfg.sh)

        # * Set learning rates
        gaussians = self.cuda_module.get_gaussians()
        gaussians.lr_rotation.fill_(cfg.lr_rotation_init)
        gaussians.lr_scale.fill_(cfg.lr_scale_init)
        gaussians.lr_mean.fill_(cfg.lr_mean_init)
        gaussians.lr_opacity.fill_(cfg.lr_opacity_init)
        gaussians.lr_channels.fill_(cfg.lr_channels)
        gaussians.lr_sh_dc.fill_(cfg.lr_sh_dc_init)
        gaussians.lr_sh_rest.fill_(cfg.lr_sh_rest)

        # * Set Adam parameters
        gaussians.beta_1.fill_(cfg.beta_1)
        gaussians.beta_2.fill_(cfg.beta_2)
        gaussians.epsilon.fill_(cfg.epsilon)
        gaussians.sh_update_laziness.fill_(cfg.sh_update_laziness)

        # * Setup MLP
        num_channels = self.cuda_module.get_num_channels()
        if num_channels != 3:
            assert cfg.post_mlp, (
                "Post-processing MLP must be enabled if the number of output channels is not 3"
            )
        if cfg.pre_mlp:
            self.pre_mlp = PreMLP(cfg, gaussians).cuda()
        if cfg.post_mlp:
            self.post_mlp = PostMLP(cfg, num_channels).cuda()

        # * Last outputs, kept for backward pass
        self.output_channels = None

    def init_exposure_comp(self, scene_info: SceneInfo):
        self.exposure_comp = ExposureComp(self.cfg, scene_info)

    def __call__(self, cam_info: CameraInfo, znear=0.0, zfar=99999.9, skip_copy=False):
        "Render the scene and takes an optimization step if a target is provided."

        # * Set camera parameters
        camera = self.cuda_module.get_camera()
        camera.znear.fill_(znear)
        camera.zfar.fill_(zfar)
        self.cuda_module.get_config().rays_from_python.fill_(False)
        camera.vertical_fov_radians.fill_(cam_info.fov_y)
        camera.set_pose(cam_info.origin_cuda(), cam_info.rotation_c2w_blender_cuda())

        # * Set gaussian colors from view direction MLP
        if self.cfg.pre_mlp:
            self.pre_mlp(cam_info)

        # * Render and step if required
        framebuffer = self.cuda_module.get_framebuffer()
        self.cuda_module.forward_pass()
        grad_enabled = torch.is_grad_enabled()
        assert not (skip_copy and grad_enabled), "skip_copy=True is not supported with gradients enabled"

        output_channels = framebuffer.output_channels.detach()
        if not skip_copy or grad_enabled:
            output_channels = output_channels.clone()
        output_channels = output_channels.moveaxis(-1, 0)

        if grad_enabled:
            assert self.output_channels is None, (
                "Called the forward pass multiple times without a backward pass"
            )
            output_channels.requires_grad_()
            self.output_channels = output_channels

        # * Apply post-processing MLP
        if self.cfg.post_mlp:
            ray_direction = framebuffer.ray_direction.detach().moveaxis(-1, 0)
            depth = framebuffer.output_depth.detach().moveaxis(-1, 0)
            hit_point = cam_info.origin_cuda()[:, None, None] + depth * ray_direction
            render = self.post_mlp(output_channels, hit_point, ray_direction)
        else:
            render = output_channels

        return render

    def backward(self, loss):
        # * Backprop from loss to raytracer (and other parameters forming the loss)
        loss.backward()
        with torch.no_grad():
            framebuffer = self.cuda_module.get_framebuffer()
            framebuffer.grad_output_channels.copy_(self.output_channels.grad.moveaxis(0, -1))
            self.output_channels = None

        # * Backprop raytracer
        self.cuda_module.backward_pass()

    def step(self):
        # * Update stats
        self.update_pruning_stats()

        # * Optimization steps
        if self.cfg.pre_mlp:
            self.pre_mlp.step()
        self.cuda_module.step()
        self.cuda_module.update_bvh()
        if self.cfg.post_mlp:
            self.post_mlp.step()
        if self.cfg.exposure_comp_enabled:
            self.exposure_comp.step()

    @staticmethod
    def from_point_cloud(
        cfg: RaytracerConfig,
        point_cloud: BasicPointCloud,
        image_width: int,
        image_height: int,
        inference_only: bool = False,
    ):
        print(f"Initializing {point_cloud.points.shape[0]} points")
        if cfg.init_binning:
            points = torch.from_numpy(point_cloud.points).cuda()
            colors = torch.from_numpy(point_cloud.colors).cuda()
            distances = torch.from_numpy(point_cloud.distances_to_cam).cuda().unsqueeze(1)

            rounded_points = (points / (cfg.init_bin_size * point_cloud.radius)).round()
            unique_pts, inverse, counts = rounded_points.unique(
                dim=0, sorted=True, return_counts=True, return_inverse=True
            )
            num_points = unique_pts.shape[0]

            color_sums = torch.zeros((num_points, 3), device=colors.device)
            color_sums.index_add_(0, inverse, colors)

            position_sums = torch.zeros((num_points, 3), device=points.device)
            position_sums.index_add_(0, inverse, points)

            distance_sums = torch.zeros((num_points, 1), device=points.device)
            distance_sums.index_add_(0, inverse, distances)

            avg_colors = color_sums / counts.unsqueeze(1)
            avg_positions = position_sums / counts.unsqueeze(1)
            avg_distances = distance_sums / counts.unsqueeze(1)

            point_cloud = BasicPointCloud(
                points=avg_positions.cpu().numpy(),
                colors=avg_colors.cpu().numpy(),
                distances_to_cam=avg_distances.squeeze(1).cpu().numpy(),
                radius=point_cloud.radius,
                normals=None,
            )
            print(f"Binning down to {num_points} points")

        torch.cuda.synchronize()  # *** Important for some reason

        num_points = point_cloud.points.shape[0]
        num_orig_points = num_points

        raytracer = Raytracer(
            cfg,
            num_points,
            image_width,
            image_height,
            inference_only=inference_only,
        )
        gaussians = raytracer.cuda_module.get_gaussians()

        rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_points, 1)
        scale = (
            torch.from_numpy(point_cloud.distances_to_cam * cfg.init_scale)
            .cuda()
            .unsqueeze(1)
            .log()
        )
        mean = torch.from_numpy(point_cloud.points).cuda()
        opacity = torch.logit(cfg.init_opacity * torch.ones(num_points, 1))
        channels = torch.cat(
            [
                torch.from_numpy(point_cloud.colors).cuda(),
                torch.randn(num_orig_points, raytracer.cuda_module.get_num_channels() - 3).cuda()
                / 2
                + 0.5,
            ],
            dim=1,
        )
        sh_coeffs_dc = (
            torch.from_numpy(point_cloud.colors).cuda().unsqueeze(1) - 0.5
        ) / 0.28209479177387814

        gaussians.rotation.copy_(rotation)
        gaussians.scale.copy_(scale)
        gaussians.mean.copy_(mean)
        gaussians.opacity.copy_(opacity)
        gaussians.channels.copy_(channels)
        gaussians.sh_coeffs_dc.copy_(sh_coeffs_dc)
        raytracer.cuda_module.rebuild_bvh()

        if cfg.pre_mlp:
            raytracer.pre_mlp.initialize()

        return raytracer

    @torch.no_grad()
    def prune(self, iteration: int, mask: Optional[torch.Tensor] = None):
        gaussians = self.cuda_module.get_gaussians()

        mean = gaussians.mean.clone()
        channels = gaussians.channels.clone()
        opacity = gaussians.opacity.clone()
        rotation = gaussians.rotation.clone()
        scale = gaussians.scale.clone()
        sh_coeffs_dc = gaussians.sh_coeffs_dc.clone()
        sh_coeffs_rest = gaussians.sh_coeffs_rest.clone()

        first_moment_mean = gaussians.first_moment_mean.clone()
        first_moment_rotation = gaussians.first_moment_rotation.clone()
        first_moment_scale = gaussians.first_moment_scale.clone()
        first_moment_opacity = gaussians.first_moment_opacity.clone()
        first_moment_channels = gaussians.first_moment_channels.clone()
        first_moment_sh_coeffs_dc = gaussians.first_moment_sh_coeffs_dc.clone()
        first_moment_sh_coeffs_rest = gaussians.first_moment_sh_coeffs_rest.clone()

        second_moment_mean = gaussians.second_moment_mean.clone()
        second_moment_rotation = gaussians.second_moment_rotation.clone()
        second_moment_scale = gaussians.second_moment_scale.clone()
        second_moment_opacity = gaussians.second_moment_opacity.clone()
        second_moment_channels = gaussians.second_moment_channels.clone()
        second_moment_sh_coeffs_dc = gaussians.second_moment_sh_coeffs_dc.clone()
        second_moment_sh_coeffs_rest = gaussians.second_moment_sh_coeffs_rest.clone()

        if mask is None:
            denom = gaussians.pruning_counter.squeeze(1).clamp(min=1)
            average_weight = gaussians.pruning_weight.squeeze(1) / denom
            mask = average_weight >= self.cfg.pruning_min_weight

        self.cuda_module.resize(mask.sum().item())

        gaussians.mean.copy_(mean[mask])
        gaussians.channels.copy_(channels[mask])
        gaussians.opacity.copy_(opacity[mask])
        gaussians.rotation.copy_(rotation[mask])
        gaussians.scale.copy_(scale[mask])
        gaussians.sh_coeffs_dc.copy_(sh_coeffs_dc[mask])
        gaussians.sh_coeffs_rest.copy_(sh_coeffs_rest[mask])

        gaussians.first_moment_mean.copy_(first_moment_mean[mask])
        gaussians.first_moment_rotation.copy_(first_moment_rotation[mask])
        gaussians.first_moment_scale.copy_(first_moment_scale[mask])
        gaussians.first_moment_opacity.copy_(first_moment_opacity[mask])
        gaussians.first_moment_channels.copy_(first_moment_channels[mask])
        gaussians.first_moment_sh_coeffs_dc.copy_(first_moment_sh_coeffs_dc[mask])
        gaussians.first_moment_sh_coeffs_rest.copy_(first_moment_sh_coeffs_rest[mask])

        gaussians.second_moment_mean.copy_(second_moment_mean[mask])
        gaussians.second_moment_rotation.copy_(second_moment_rotation[mask])
        gaussians.second_moment_scale.copy_(second_moment_scale[mask])
        gaussians.second_moment_opacity.copy_(second_moment_opacity[mask])
        gaussians.second_moment_channels.copy_(second_moment_channels[mask])
        gaussians.second_moment_sh_coeffs_dc.copy_(second_moment_sh_coeffs_dc[mask])
        gaussians.second_moment_sh_coeffs_rest.copy_(second_moment_sh_coeffs_rest[mask])

        gaussians.pruning_weight.zero_()
        gaussians.pruning_counter.zero_()

        if self.cfg.pre_mlp:
            self.pre_mlp.prune(mask)

    @torch.no_grad()
    def update_pruning_stats(self):
        gaussians = self.cuda_module.get_gaussians()
        mask = gaussians.was_visible.squeeze(1)

        gaussians.pruning_counter[mask] += 1

    @staticmethod
    def from_safetensors(
        cfg: RaytracerConfig,
        path: str,
        image_width: int,
        image_height: int,
        inference_only: bool = False,
    ):
        state_dict = safetensors.torch.load_file(path)
        del state_dict["image_width"]
        del state_dict["image_height"]

        # * Older checkpoints may include bg_color, but config is now the source of truth.
        state_dict.pop("bg_color", None)

        num_points = state_dict["mean"].shape[0]
        raytracer = Raytracer(
            cfg,
            num_points,
            image_width,
            image_height,
            inference_only=inference_only,
        )
        gaussians = raytracer.cuda_module.get_gaussians()
        gaussians.mean.copy_(state_dict["mean"])
        gaussians.rotation.copy_(state_dict["rotation"])
        gaussians.scale.copy_(state_dict["scale"])
        gaussians.opacity.copy_(state_dict["opacity"])
        gaussians.channels.copy_(state_dict["channels"])
        gaussians.sh_coeffs_dc.copy_(state_dict["sh_coeffs_dc"])
        if cfg.sh:
            gaussians.sh_coeffs_rest.copy_(state_dict["sh_coeffs_rest"])
        gaussians.current_sh_degree.copy_(state_dict["current_sh_degree"])
        raytracer.cuda_module.rebuild_bvh()

        del state_dict["mean"]
        del state_dict["rotation"]
        del state_dict["scale"]
        del state_dict["opacity"]
        del state_dict["channels"]
        del state_dict["sh_coeffs_dc"]
        del state_dict["sh_coeffs_rest"]
        del state_dict["current_sh_degree"]

        if cfg.pre_mlp:
            raytracer.pre_mlp.initialize()

        raytracer.load_state_dict(state_dict)
        return raytracer

    def save_safetensors(self, model_path: str, iteration: int):
        gaussians = self.cuda_module.get_gaussians()
        tensors = {
            "mean": gaussians.mean,
            "rotation": gaussians.rotation,
            "scale": gaussians.scale,
            "opacity": gaussians.opacity,
            "channels": gaussians.channels,
            "sh_coeffs_dc": gaussians.sh_coeffs_dc,
            "sh_coeffs_rest": gaussians.sh_coeffs_rest,
            "current_sh_degree": gaussians.current_sh_degree,
            "image_width": torch.tensor(self.image_width),
            "image_height": torch.tensor(self.image_height),
        }

        path = os.path.join(model_path, f"gaussians_{iteration:05d}.safetensors")
        safetensors.torch.save_file({**self.state_dict(), **tensors}, path)

    @staticmethod
    def from_ply(
        cfg: RaytracerConfig,
        path: str,
        image_width: int,
        image_height: int,
        inference_only: bool = False,
    ):
        plydata = PlyData.read(path)
        vertex = plydata["vertex"].data

        points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
        opacities = np.expand_dims(vertex["opacity"], axis=1).astype(np.float32)
        rotations = np.stack(
            [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1
        ).astype(np.float32)
        scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1).astype(
            np.float32
        )

        raytracer = Raytracer(
            cfg,
            points.shape[0],
            image_width,
            image_height,
            inference_only=inference_only,
        )
        gaussians = raytracer.cuda_module.get_gaussians()
        gaussians.mean.copy_(torch.from_numpy(points))
        gaussians.opacity.copy_(torch.from_numpy(opacities))
        gaussians.rotation.copy_(torch.from_numpy(rotations))
        gaussians.scale.copy_(torch.from_numpy(scales))

        if cfg.sh:
            sh_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(
                np.float32
            )
            sh_dc = sh_dc[:, None, :]
            f_rest_keys = [k for k in vertex.dtype.fields if k.startswith("f_rest_")]
            sh_rest = (
                np.stack([vertex[k] for k in f_rest_keys], axis=1)
                .astype(np.float32)
                .reshape(points.shape[0], -1, 3)
            )
            gaussians.sh_coeffs_dc.copy_(torch.from_numpy(sh_dc))
            gaussians.sh_coeffs_rest.copy_(torch.from_numpy(sh_rest))
            gaussians.current_sh_degree.fill_(cfg.sh_max_degree)
            gaussians.channels.zero_()
        else:
            channels = np.stack(
                [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1
            ).astype(np.float32)
            gaussians.channels.copy_(torch.from_numpy(channels))

        raytracer.cuda_module.rebuild_bvh()
        return raytracer

    def save_ply(self, model_path: str, iteration: int):
        """
        Save the current state of the raytracer's gaussians to a PLY file,
        including SH rest coefficients following the 3DGS convention.
        """
        gaussians = self.cuda_module.get_gaussians()
        points = gaussians.mean.cpu().numpy()
        channels = gaussians.channels.cpu().numpy()
        opacities = gaussians.opacity.cpu().numpy()
        rotations = gaussians.rotation.cpu().numpy()
        scales = gaussians.scale.cpu().numpy()
        if self.cfg.sh:
            sh_dc = gaussians.sh_coeffs_dc.cpu().numpy()
            sh_rest = gaussians.sh_coeffs_rest.cpu().numpy()

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
        for i in range(sh_rest.shape[1] * sh_rest.shape[2]):
            dtype_fields.append((f"f_rest_{i}", "f4"))

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
        if self.cfg.sh:
            vertex["f_dc_0"] = sh_dc[:, 0, 0]
            vertex["f_dc_1"] = sh_dc[:, 0, 1]
            vertex["f_dc_2"] = sh_dc[:, 0, 2]
            for i in range(sh_rest.shape[1]):
                vertex[f"f_rest_{3 * i}"] = sh_rest[:, i, 0]
                vertex[f"f_rest_{3 * i + 1}"] = sh_rest[:, i, 1]
                vertex[f"f_rest_{3 * i + 2}"] = sh_rest[:, i, 2]
        else:
            vertex["f_dc_0"] = channels[:, 0]
            vertex["f_dc_1"] = channels[:, 1]
            vertex["f_dc_2"] = channels[:, 2]

        ply_element = PlyElement.describe(vertex, "vertex")
        PlyData([ply_element]).write(f"{model_path}/gaussians_{iteration:05d}.ply")
