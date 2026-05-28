import json
import os
from typing import Optional

import numpy as np
from gray.camera import CameraInfo


SAVED_VIEWS_FILENAME = "viewer_saved_cameras.json"


def camera_storage_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return os.path.dirname(path)
    return path


def saved_view_file(path: Optional[str]) -> Optional[str]:
    storage_dir = camera_storage_dir(path)
    if storage_dir is None:
        return None
    return os.path.join(storage_dir, SAVED_VIEWS_FILENAME)


def _camera_info_from_saved_json(data: dict) -> CameraInfo:
    if "R" in data or "origin" in data:
        return CameraInfo.from_json(data)

    to_world = np.asarray(data["to_world"], dtype=np.float32)
    if to_world.shape != (4, 4):
        raise ValueError("saved camera to_world must be a 4x4 matrix")

    return CameraInfo(
        uid=0,
        R=to_world[:3, :3],
        T=np.zeros(3, dtype=np.float32),
        origin=to_world[:3, 3],
        fov_y=float(data["fov_y"]),
        fov_x=float(data.get("fov_x", data["fov_y"])),
        image_path="",
        image_name=str(data["name"]).strip(),
        image_width=0,
        image_height=0,
        is_test=False,
    )


def load_saved_views(path: Optional[str]) -> list[CameraInfo]:
    if path is None or not os.path.exists(path):
        return []

    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except OSError as exc:
        print(f"WARNING: Failed to open saved cameras at '{path}': {exc}")
        return []
    except json.JSONDecodeError as exc:
        print(f"WARNING: Failed to parse saved cameras at '{path}': {exc}")
        return []

    if not isinstance(payload, list):
        print(f"WARNING: Expected a list of cameras in '{path}', got {type(payload).__name__}")
        return []

    cameras = []
    for idx, camera_data in enumerate(payload):
        if not isinstance(camera_data, dict):
            print(f"WARNING: Skipping saved camera #{idx} in '{path}': expected an object")
            continue
        try:
            cameras.append(_camera_info_from_saved_json(camera_data))
        except (KeyError, TypeError, ValueError) as exc:
            print(f"WARNING: Skipping saved camera #{idx} in '{path}': {exc}")
    return cameras


def write_saved_views(path: str, views: list[CameraInfo]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([view.to_json() for view in views], f, indent=4)


def upsert_saved_view(views: list[CameraInfo], saved_view: CameraInfo) -> list[CameraInfo]:
    updated_views = list(views)
    for idx, view in enumerate(updated_views):
        if view.image_name == saved_view.image_name:
            updated_views[idx] = saved_view
            break
    else:
        updated_views.append(saved_view)
    return updated_views
