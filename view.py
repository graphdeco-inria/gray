#
# This file is licensed under the Apache 2.0 license in viewer/LICENSE.md.
#


import os
from threading import Lock
from argparse import ArgumentParser
from imgui_bundle import imgui_ctx, imgui
from gray.camera import CameraInfo
from viewer import Viewer
from viewer.types import ViewerMode
from viewer.widgets.image import TorchImage
from viewer.widgets.cameras.fps import FPSCamera
from viewer.widgets.monitor import PerformanceMonitor

from dataclasses import dataclass
import tyro
from tyro.conf import subcommand, arg
from typing import Annotated, List, Optional
import json


@dataclass
class ViewerCLI:
    model_path: Annotated[Optional[str], arg(aliases=["-m"])] = None
    iteration: Annotated[int, arg(aliases=["-t"])] = -1
    server: bool = False # * Run as a WebSocket server on this (GPU) machine
    client: Optional[str] = None # * Connect as a client to the given server IP address
    port: int = 6009
    image_scale: Annotated[float, arg(aliases=["-x"])] = 1.0  # * Image scale factor; >1 = upsampling, <1 = downsampling


class GaussianViewer(Viewer):
    def __init__(
        self,
        raytracer: "Raytracer",
        train_cameras: List["CameraInfo"],
        test_cameras: Optional[List["CameraInfo"]],
        training=False,
        mode: ViewerMode = ViewerMode.LOCAL,
        image_scale: float = 1.0,
    ):
        super().__init__(mode)
        self.window_title = "Gaussian Viewer"
        self.gaussian_lock = Lock()
        self.raytracer = raytracer
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras
        self.must_rebuild_bvh = False
        self.training = training
        self.image_scale = image_scale

    def import_server_modules(self):
        global torch
        import torch

        global gray
        import gray

    def create_widgets(self):
        # * Dummy fallback intrinsics before real cameras arrive from dataset/server
        self.camera_widget = FPSCamera(self.mode, 1, 1, 30, 0.001, 100)
        init_camera = (self.test_cameras or [None])[0] or (self.train_cameras or [None])[0]
        if init_camera is not None:
            self.camera_widget.res_x = max(1, round(init_camera.image_width * self.image_scale))
            self.camera_widget.res_y = max(1, round(init_camera.image_height * self.image_scale))
            self.camera_widget.compute_fov_x()
            self.camera_widget.set(init_camera)

        # * On the server, ignore the client's requested resolution and keep our own
        _fixed_res_x = self.camera_widget.res_x
        _fixed_res_y = self.camera_widget.res_y
        _orig_cam_server_recv = self.camera_widget.server_recv
        def _server_recv_fixed_res(binary, text):
            _orig_cam_server_recv(binary, text)
            self.camera_widget.res_x = _fixed_res_x
            self.camera_widget.res_y = _fixed_res_y
            self.camera_widget.compute_fov_x()
        self.camera_widget.server_recv = _server_recv_fixed_res
        self.point_view = TorchImage(self.mode)

        # * Render modes
        self.render_modes = ["Gaussians"]
        if self.raytracer is not None:
            if self.raytracer.cfg.render_depth:
                self.render_modes.append("Depth")
            if not self.training and not self.raytracer.cfg.post_mlp:
                self.render_modes.append("Ellipsoids")
        self.render_mode = "Gaussians"

        # * Render settings
        self.scaling_modifier = 1.0
        # z-near for the renderer
        self.znear = 0.0
        # Depth display remap range
        self.depth_min = 0.0
        self.depth_max = 10.0
        self.ellipsoid_min_opacity = 0.0

        # * Camera view
        self.current_train_cam = -1
        self.current_test_cam = -1

    def step(self):
        camera = CameraInfo(
            uid=0,
            R=self.camera_widget.to_world[:3, :3],
            T=None,
            origin=self.camera_widget.origin,
            fov_x=self.camera_widget.fov_x,
            fov_y=self.camera_widget.fov_y,
            image_path=None,
            image_name=None,
            image_width=self.camera_widget.res_x,
            image_height=self.camera_widget.res_y,
            is_test=False,
        )

        if self.render_mode in ["Gaussians", "Depth", "Ellipsoids"]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                with self.gaussian_lock:
                    config = self.raytracer.cuda_module.get_config()
                    config.global_scale_factor.copy_(self.scaling_modifier)
                    config.render_ellipsoids.fill_(self.render_mode == "Ellipsoids")
                    config.ellipsoid_min_opacity.fill_(self.ellipsoid_min_opacity)
                    if self.must_rebuild_bvh:
                        self.raytracer.cuda_module.rebuild_bvh()
                        self.must_rebuild_bvh = False

                    render = self.raytracer(
                        camera,
                        znear=self.znear,
                    ).clamp(0, 1)

                if self.render_mode in ["Gaussians", "Ellipsoids"]:
                    net_image = render.moveaxis(0, -1)
                else:
                    framebuffer = self.raytracer.cuda_module.get_framebuffer()
                    net_image = process_depth_map(
                        framebuffer.output_depth.detach(),
                        depth_min=self.depth_min,
                        depth_max=self.depth_max,
                    )
            end.record()
            end.synchronize()
            self.point_view.step(net_image)
            render_time = start.elapsed_time(end)

    def show_gui(self):
        with imgui_ctx.begin(f"Point View Settings"):
            _, render_mode_choice = imgui.list_box(
                "Render Mode", self.render_modes.index(self.render_mode), self.render_modes
            )
            self.render_mode = self.render_modes[render_mode_choice]

            imgui.separator_text("Render Settings")
            if self.render_mode in ["Gaussians", "Depth", "Ellipsoids"]:
                scaling_changed, self.scaling_modifier = imgui.drag_float(
                    "Scaling Modifier", self.scaling_modifier, v_min=0, v_max=2, v_speed=0.01
                )
                if scaling_changed:
                    self.must_rebuild_bvh = True
                if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    self.scaling_modifier = 1.0

                znear_changed, self.znear = imgui.drag_float(
                    "Z Near", self.znear, v_min=0.0, v_max=1000.0, v_speed=0.01
                )
                if znear_changed:
                    # No BVH rebuild required, just update camera param next render
                    pass

                if self.render_mode == "Depth":
                    depth_min_changed, self.depth_min = imgui.drag_float(
                        "Depth Min",
                        self.depth_min,
                        v_min=-100000.0,
                        v_max=100000.0,
                        v_speed=0.1,
                    )
                    if depth_min_changed:
                        self.depth_min = min(self.depth_min, self.depth_max - 1e-3)
                    if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                        self.depth_min = 0.0

                    depth_max_changed, self.depth_max = imgui.drag_float(
                        "Depth Max",
                        self.depth_max,
                        v_min=0.001,
                        v_max=100000.0,
                        v_speed=0.1,
                    )
                    if depth_max_changed:
                        self.depth_max = max(self.depth_max, self.depth_min + 1e-3)
                    if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                        self.depth_max = 10.0

                if self.render_mode == "Ellipsoids":
                    _, self.ellipsoid_min_opacity = imgui.slider_float(
                        "Min Opacity", self.ellipsoid_min_opacity, v_min=0.0, v_max=1.0
                    )
                    if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.right):
                        self.ellipsoid_min_opacity = 0.0

            imgui.separator_text("Camera Settings")
            self.camera_widget.show_gui()

            using_train_cam = self.current_train_cam != -1
            if not using_train_cam:
                imgui.push_style_color(imgui.Col_.frame_bg, (0.0, 0.0, 0.0, 0.0))
                imgui.push_style_color(imgui.Col_.frame_bg_hovered, (0.0, 0.0, 0.0, 0.0))
                imgui.push_style_color(imgui.Col_.frame_bg_active, (0.0, 0.0, 0.0, 0.0))
            train_cam_changed, self.current_train_cam = imgui.input_int(
                "Set Train View", self.current_train_cam, step=1, step_fast=10
            )
            if (
                not train_cam_changed
                and imgui.is_item_hovered()
                and imgui.is_mouse_clicked(imgui.MouseButton_.right)
            ):
                train_cam_changed = True
                self.current_train_cam = 0
            if not using_train_cam:
                imgui.pop_style_color(3)
            self.current_train_cam = max(
                -1, min(len(self.train_cameras) - 1 if self.train_cameras else -1, self.current_train_cam)
            )

            using_test_cam = self.current_test_cam != -1
            if not using_test_cam:
                imgui.push_style_color(imgui.Col_.frame_bg, (0.0, 0.0, 0.0, 0.0))
                imgui.push_style_color(imgui.Col_.frame_bg_hovered, (0.0, 0.0, 0.0, 0.0))
                imgui.push_style_color(imgui.Col_.frame_bg_active, (0.0, 0.0, 0.0, 0.0))
            test_cam_changed, self.current_test_cam = imgui.input_int(
                "Set Test View", self.current_test_cam, step=1, step_fast=10
            )
            if (
                not test_cam_changed
                and imgui.is_item_hovered()
                and imgui.is_mouse_clicked(imgui.MouseButton_.right)
            ):
                test_cam_changed = True
                self.current_test_cam = 0
            if not using_test_cam:
                imgui.pop_style_color(3)
            self.current_test_cam = max(-1, min(len(self.test_cameras) - 1 if self.test_cameras else -1, self.current_test_cam))

            if train_cam_changed and self.train_cameras:
                self.camera_widget.set(self.train_cameras[self.current_train_cam])
                self.current_test_cam = -1
            elif test_cam_changed and self.test_cameras:
                self.camera_widget.set(self.test_cameras[self.current_test_cam])
                self.current_train_cam = -1

        with imgui_ctx.begin("Point View"):
            self.point_view.show_gui()

            if imgui.is_item_hovered():
                self.camera_widget.process_mouse_input()

            if imgui.is_item_focused() or imgui.is_item_hovered():
                self.camera_widget.process_keyboard_input()

    def client_send(self):
        return None, {
            "scaling_modifier": self.scaling_modifier,
            "render_mode": self.render_mode,
            "znear": float(self.znear),
            "depth_min": float(self.depth_min),
            "depth_max": float(self.depth_max),
            "ellipsoid_min_opacity": float(self.ellipsoid_min_opacity),
        }

    def onconnect(self, _):
        self._cameras_sent = False

    def server_send(self):
        text = {"render_modes": self.render_modes}
        # * Send the cameras list once per connection so the client can control cameras
        if not getattr(self, "_cameras_sent", False):
            text["train_cameras"] = [c.to_json() for c in self.train_cameras]
            text["test_cameras"] = [c.to_json() for c in (self.test_cameras or [])]
            self._cameras_sent = True
        return None, text

    def client_recv(self, _, text):
        if not text:
            return
        if "render_modes" in text:
            self.render_modes = text["render_modes"]
            if self.render_mode not in self.render_modes:
                self.render_mode = self.render_modes[0]
        if "train_cameras" in text:
            self.train_cameras = [CameraInfo.from_json(c) for c in text["train_cameras"]]
            self.test_cameras = [CameraInfo.from_json(c) for c in text["test_cameras"]]
            init_cam = self.test_cameras[0] if self.test_cameras else (self.train_cameras[0] if self.train_cameras else None)
            if init_cam is not None:
                self.camera_widget.res_x = init_cam.image_width
                self.camera_widget.res_y = init_cam.image_height
                self.camera_widget.compute_fov_x()
                self.camera_widget.set(init_cam)

    def show_status(self):
        imgui.text(f"{self.camera_widget.res_x} × {self.camera_widget.res_y}")

    def server_recv(self, _, text):
        new_scaling = text["scaling_modifier"]
        if new_scaling != self.scaling_modifier:
            self.must_rebuild_bvh = True
        self.scaling_modifier = new_scaling
        self.render_mode = text["render_mode"]
        if "znear" in text:
            self.znear = float(text["znear"])
        if "depth_min" in text:
            self.depth_min = float(text["depth_min"])
        if "depth_max" in text:
            self.depth_max = float(text["depth_max"])
        if "ellipsoid_min_opacity" in text:
            self.ellipsoid_min_opacity = float(text["ellipsoid_min_opacity"])


def process_depth_map(raw_depth, depth_min: float, depth_max: float):
    """Normalize depth and apply a viridis colormap; keep misses black."""
    import torch

    # * Normalize depth so depth_min/near -> white (1) and depth_max/far -> black (0).
    depth_max = max(depth_max, depth_min + 1e-6)
    depth_range = depth_max - depth_min
    depth = torch.clamp((depth_max - raw_depth) / depth_range, 0.0, 1.0)

    # * Piecewise-linear viridis mapping on GPU.
    v = depth[..., 0]
    flat_v = v.reshape(-1)
    knots = raw_depth.new_tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    colors = raw_depth.new_tensor(
        [
            [0.267004, 0.004874, 0.329415],
            [0.253935, 0.265254, 0.529983],
            [0.163625, 0.471133, 0.558148],
            [0.134692, 0.658636, 0.517649],
            [0.477504, 0.821444, 0.318195],
            [0.993248, 0.906157, 0.143936],
        ]
    )

    idx = torch.bucketize(flat_v, knots, right=True) - 1
    idx = torch.clamp(idx, 0, knots.numel() - 2)
    left_knot = knots[idx]
    right_knot = knots[idx + 1]
    t = ((flat_v - left_knot) / (right_knot - left_knot)).unsqueeze(-1)
    left_color = colors[idx]
    right_color = colors[idx + 1]
    depth_rgb = (left_color + t * (right_color - left_color)).view(v.shape[0], v.shape[1], 3)

    hit_mask = raw_depth > 0.0
    return torch.where(hit_mask.expand_as(depth_rgb), depth_rgb, torch.zeros_like(depth_rgb))


if __name__ == "__main__":
    cli, unknown_args = tyro.cli(ViewerCLI, return_unknown_args=True)

    if cli.client is not None:
        # * Client mode: no model loading, just connect to the server
        viewer = GaussianViewer(None, [], [], mode=ViewerMode.CLIENT)
        viewer.run(ip=cli.client, port=cli.port)
    else:
        if cli.model_path is None:
            import sys
            print("error: -m/--model-path is required unless --client <IP> is specified", file=sys.stderr)
            sys.exit(1)

        # * Defer loading slower modules after CLI parsing
        from gray.prelude import Config, search_for_max_iteration, Raytracer, CameraInfo

        # * Load the config from JSON and allow for Config overrides
        saved_cli_path = os.path.join(cli.model_path, "config.json")
        cfg = tyro.cli(
            Config, args=unknown_args, default=Config(**json.load(open(saved_cli_path, "r")))
        )

        # * Make it possible to point directly to a gaussians file
        if cli.model_path.endswith(".safetensors"):
            iteration = cfg.iteration
            save_path = cli.model_path
        elif cli.iteration != -1:
            iteration = cli.iteration
            save_path = os.path.join(cli.model_path, f"gaussians_{iteration:05d}.safetensors")
        else:
            iteration = search_for_max_iteration(cli.model_path)
            save_path = os.path.join(cli.model_path, f"gaussians_{iteration:05d}.safetensors")

        # * Load the cameras and raytracer
        cameras = [
            CameraInfo.from_json(x)
            for x in json.load(open(os.path.join(cli.model_path, "cameras.json"), "r"))
        ]
        image_width = max(1, round(cameras[0].image_width * cli.image_scale))
        image_height = max(1, round(cameras[0].image_height * cli.image_scale))
        cfg.render_depth = True  # Always enable depth output for the viewer
        raytracer = Raytracer.from_safetensors(cfg, save_path, image_width, image_height)
        train_cameras = [c for c in cameras if not c.is_test]
        test_cameras = [c for c in cameras if c.is_test]

        mode = ViewerMode.SERVER if cli.server else ViewerMode.LOCAL
        viewer = GaussianViewer(raytracer, train_cameras, test_cameras, mode=mode, image_scale=cli.image_scale)
        viewer.run(ip="0.0.0.0" if cli.server else "localhost", port=cli.port)
