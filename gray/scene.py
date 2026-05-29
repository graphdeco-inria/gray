from __future__ import annotations

from gray.imports import *
from gray.utils import *
from gray.config import Config
from gray.camera import CameraInfo
import gray.colmap as colmap

from torchvision.io import read_image, ImageReadMode
from concurrent.futures import ThreadPoolExecutor
import torch
import struct

executor = ThreadPoolExecutor()


@dataclass
class BasicPointCloud:
    points: np.array
    colors: np.array
    radius: float
    normals: Optional[np.array] = None
    distances_to_cam: Optional[np.array] = None


@dataclass
class SceneInfo:
    point_cloud: BasicPointCloud
    train_cameras: List[CameraInfo]
    test_cameras: List[CameraInfo]
    train_images: List[torch.Tensor]
    test_images: List[torch.Tensor]
    pc_path: str
    is_nerf_synthetic: bool

    @staticmethod
    def from_colmap(cfg: Config, llffhold=8, parse_point_cloud=True) -> SceneInfo:
        path = cfg.source_path

        # * Read colmap data
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            cam_extrinsics = colmap.read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = colmap.read_intrinsics_binary(cameras_intrinsic_file)
        except FileNotFoundError:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = colmap.read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = colmap.read_intrinsics_text(cameras_intrinsic_file)

        # * Select views for eval
        if cfg.eval:
            if "360" in path:
                llffhold = 8
            if llffhold:
                print("------------LLFF HOLD-------------")
                cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
                cam_names = sorted(cam_names)
                test_cam_names_list = [
                    name for idx, name in enumerate(cam_names) if idx % llffhold == 0
                ]
            else:
                with open(os.path.join(path, "sparse/0", "test.txt"), "r") as file:
                    test_cam_names_list = [line.strip() for line in file]
        else:
            test_cam_names_list = []

        # * Parse cameras
        cam_infos_unsorted = []
        for key in cam_extrinsics:
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            cam_info = CameraInfo.from_colmap(
                cfg, key, extr, intr, extr.name in test_cam_names_list
            )
            cam_infos_unsorted.append(cam_info)
        cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
        train_cam_infos = [c for c in cam_infos if not c.is_test]
        test_cam_infos = [c for c in cam_infos if c.is_test]
        radius = get_nerf_pp_norm(train_cam_infos)["radius"]

        # * Parse point cloud, cache to safetensors for fast loading
        if parse_point_cloud:
            if os.path.isabs(cfg.point_cloud_file):
                pc_path = cfg.point_cloud_file
            else:
                pc_path = os.path.join(path, cfg.point_cloud_file)
            safetensor_path = pc_path.replace(".ply", ".safetensors")
            import safetensors.numpy

            if os.path.exists(safetensor_path) and (
                not os.path.exists(pc_path)
                or os.path.getmtime(safetensor_path) >= os.path.getmtime(pc_path)
            ):
                data = safetensors.numpy.load_file(safetensor_path)
                positions = data["positions"]
                colors = data["colors"]
                if "distances_to_cam" in data:
                    distances_to_cam = data["distances_to_cam"]
                else:
                    distances_to_cam = None
            else:
                plydata = PlyData.read(pc_path)
                vertices = plydata["vertex"]
                positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
                colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
                distances_to_cam = None
                safetensors.numpy.save_file(
                    {"positions": positions, "colors": colors}, safetensor_path
                )
            pcd = BasicPointCloud(
                points=positions,
                colors=colors,
                normals=None,
                radius=radius,
                distances_to_cam=distances_to_cam,
            )
            pc_path = safetensor_path
        else:
            pcd = None
            pc_path = None

        # * Parse images, read off-thread to avoid blocking
        def load_image(cam):
            image = read_image(cam.image_path, ImageReadMode.RGB).cuda() / 255
            return cam.image_name, image

        train_futures = executor.map(load_image, train_cam_infos)
        test_futures = executor.map(load_image, test_cam_infos)
        train_images = dict(train_futures)
        test_images = dict(test_futures)

        return SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            train_images=train_images,
            test_images=test_images,
            pc_path=pc_path,
            is_nerf_synthetic=False,
        )


