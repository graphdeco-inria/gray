from __future__ import annotations

from gray.imports import *
from gray.utils import *
from gray.config import Config
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
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)

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

        # * Compute background color as the average color of all training images
        all_colors = torch.cat([img.reshape(3, -1) for img in train_images.values()], dim=1)
        bg_color = tuple(all_colors.mean(dim=1).cpu().numpy().tolist())

        return SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            train_images=train_images,
            test_images=test_images,
            pc_path=pc_path,
            is_nerf_synthetic=False,
            bg_color=bg_color,
        )


@dataclass
class CameraInfo:
    uid: int
    R: np.array
    T: np.array
    origin: np.array
    fov_y: np.array
    fov_x: np.array
    image_path: str
    image_name: str
    image_width: int
    image_height: int
    is_test: bool

    @staticmethod
    def from_colmap(cfg: Config, key, extr: colmap.Image, intr: colmap.Camera, is_test: bool):
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
                "Colmap camera model not handled: only undistorted cameras (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            )

        # * Get image path and name
        if os.path.isabs(extr.name):
            image_path = extr.name
        else:
            images_dir_name = "images" if cfg.images_dir is None else cfg.images_dir
            image_path = os.path.join(cfg.source_path, images_dir_name, extr.name)
        image_name = extr.name

        # * Convert extension to .png if not already
        base, ext = os.path.splitext(image_path)
        if ext.lower() != ".png":
            image_path = base + ".png"
            image_name = os.path.splitext(image_name)[0] + ".png"

        # * Get image size, PIL is lazy and this does not decode the image
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        cam_info = CameraInfo(
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
        return cam_info

    @staticmethod
    def from_json(json_data: dict):
        fields = CameraInfo.__dataclass_fields__
        kwargs = {}
        for field in fields:
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
