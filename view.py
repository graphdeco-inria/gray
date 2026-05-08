#
# This file is licensed under the Apache 2.0 license in viewer/LICENSE.md.
#


import os
from OpenGL.GL import *
import numpy as np
from threading import Lock
from argparse import ArgumentParser
from imgui_bundle import imgui_ctx, imgui
import gray.scene
from viewer import Viewer
from viewer.types import ViewerMode
from viewer.widgets.image import TorchImage
from viewer.widgets.cameras.fps import FPSCamera
from viewer.widgets.monitor import PerformanceMonitor
from viewer.widgets.ellipsoid_viewer import EllipsoidViewer

from dataclasses import dataclass
import tyro
from tyro.conf import subcommand, arg
from typing import Annotated, List, Optional
import json


@dataclass
class ViewerCLI:
    model_path: Annotated[str, arg(aliases=["-m"])]
    iteration: Annotated[int, arg(aliases=["-t"])] = -1


class GaussianViewer(Viewer):
    def __init__(
        self,
        raytracer: "Raytracer",
        train_cameras: List["CameraInfo"],
        test_cameras: Optional[List["CameraInfo"]],
        training=False,
    ):
        super().__init__(ViewerMode.LOCAL)
        self.window_title = "Gaussian Viewer"
        self.gaussian_lock = Lock()
        self.raytracer = raytracer
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras
        self.must_rebuild_bvh = False
        self.training = training

    def import_server_modules(self):
        global torch
        import torch

        global gray
        import gray

    def create_widgets(self):
        init_camera = self.test_cameras[0] if self.test_cameras else self.train_cameras[0]
        self.camera_widget = FPSCamera(self.mode, 1297, 840, 47, 0.001, 100)
        self.camera_widget.set(init_camera)
        self.point_view = TorchImage(self.mode)
        self.ellipsoid_viewer = EllipsoidViewer(self.mode)

        # * Render modes
        self.render_modes = ["Splats"]
        if self.raytracer.cfg.render_depth:
            self.render_modes.append("Depth")
        if not self.training and not self.raytracer.cfg.post_mlp:
            self.render_modes.append("Ellipsoids")
        self.render_mode = "Splats"

        # * Render settings
        self.scaling_modifier = 1.0
        # z-near for the renderer
        self.znear = 0.0

        # * Camera view
        self.current_train_cam = -1
        self.current_test_cam = -1
        self.camera_widget.set(init_camera)

    def step(self):
        camera = gray.scene.CameraInfo(
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

        if self.ellipsoid_viewer.num_gaussians is None and not self.raytracer.cfg.post_mlp:
            gaussians = self.raytracer.cuda_module.get_gaussians()

            if self.raytracer.cfg.sh:
                colors = (gaussians.sh_coeffs_dc * 0.28209479177387814 + 0.5) / 3
            else:
                colors = gaussians.channels / 3

            self.ellipsoid_viewer.upload(
                gaussians.mean.detach().cpu().numpy(),
                gaussians.rotation.detach().cpu().numpy(),
                gaussians.scale.exp().detach().cpu().numpy(),
                gaussians.opacity.sigmoid().detach().cpu().numpy(),
                colors.detach().cpu().numpy(),
            )

        if self.render_mode in ["Splats", "Depth"]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                with self.gaussian_lock:
                    config = self.raytracer.cuda_module.get_config()
                    config.global_scale_factor.copy_(self.scaling_modifier)
                    if self.must_rebuild_bvh:
                        self.raytracer.cuda_module.rebuild_bvh()
                        self.must_rebuild_bvh = False
                    render = self.raytracer(camera, znear=self.znear).clamp(0, 1)
                if self.render_mode == "Splats":
                    net_image = render.moveaxis(0, -1)
                else:
                    framebuffer = self.raytracer.cuda_module.get_framebuffer()
                    net_image = framebuffer.output_depth.detach().clone().repeat(1, 1, 3)
                    net_image = (net_image - net_image.min()) / (net_image.max() - net_image.min())
            end.record()
            end.synchronize()
            self.point_view.step(net_image)
            render_time = start.elapsed_time(end)
        if self.render_mode == "Ellipsoids":
            self.ellipsoid_viewer.step(self.camera_widget)
            render_time = glGetQueryObjectuiv(self.ellipsoid_viewer.query, GL_QUERY_RESULT) / 1e6

    def show_gui(self):
        with imgui_ctx.begin(f"Point View Settings"):
            _, render_mode_choice = imgui.list_box(
                "Render Mode", self.render_modes.index(self.render_mode), self.render_modes
            )
            self.render_mode = self.render_modes[render_mode_choice]

            imgui.separator_text("Render Settings")
            if self.render_mode in ["Splats", "Depth"]:
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

            if self.render_mode == "Ellipsoids":
                _, self.ellipsoid_viewer.scaling_modifier = imgui.drag_float(
                    "Scaling Factor",
                    self.ellipsoid_viewer.scaling_modifier,
                    v_min=0,
                    v_max=10,
                    v_speed=0.01,
                )

                _, self.ellipsoid_viewer.render_floaters = imgui.checkbox(
                    "Render Floaters", self.ellipsoid_viewer.render_floaters
                )
                _, self.ellipsoid_viewer.limit = imgui.drag_float(
                    "Alpha Threshold", self.ellipsoid_viewer.limit, v_min=0, v_max=1, v_speed=0.01
                )

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
                -1, min(len(self.train_cameras) - 1, self.current_train_cam)
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
            self.current_test_cam = max(-1, min(len(self.test_cameras) - 1, self.current_test_cam))

            if train_cam_changed:
                self.camera_widget.set(self.train_cameras[self.current_train_cam])
                self.current_test_cam = -1
            elif test_cam_changed:
                self.camera_widget.set(self.test_cameras[self.current_test_cam])
                self.current_train_cam = -1

        with imgui_ctx.begin("Point View"):
            if self.render_mode in ["Splats", "Depth"]:
                self.point_view.show_gui()
            else:
                self.ellipsoid_viewer.show_gui()

            if imgui.is_item_hovered():
                self.camera_widget.process_mouse_input()

            if imgui.is_item_focused() or imgui.is_item_hovered():
                self.camera_widget.process_keyboard_input()

    def client_send(self):
        return None, {
            "scaling_modifier": self.scaling_modifier,
            "render_mode": self.render_mode,
            "znear": float(self.znear),
        }

    def server_recv(self, _, text):
        self.scaling_modifier = text["scaling_modifier"]
        self.render_mode = text["render_mode"]
        if "znear" in text:
            self.znear = float(text["znear"])


if __name__ == "__main__":
    cli, unknown_args = tyro.cli(ViewerCLI, return_unknown_args=True)

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
    image_width, image_height = cameras[0].image_width, cameras[0].image_height
    raytracer = Raytracer.from_safetensors(cfg, save_path, image_width, image_height)
    train_cameras = [c for c in cameras if not c.is_test]
    test_cameras = [c for c in cameras if c.is_test]
    viewer = GaussianViewer(raytracer, train_cameras, test_cameras)

    viewer.run()
