#
# This file is licensed under the Apache 2.0 license in viewer/LICENSE.md.
#

import os
from collections import Counter
from typing import Optional

import numpy as np

from gray.camera import CameraInfo
from saved_cameras import load_saved_cameras, saved_camera_file, upsert_saved_camera, write_saved_cameras

from . import Widget
from .cameras import Camera
from ..types import ViewerMode


class SavedViews(Widget):
    def __init__(
        self,
        mode: ViewerMode,
        camera_widget: Camera,
        model_path: Optional[str] = None,
        source_path: Optional[str] = None,
    ):
        self.camera_widget = camera_widget
        self.saved_view_name = ""
        self.saved_view_status = ""
        self.current_saved_view = -1
        self.saved_views: list[CameraInfo] = []
        self.saved_view_labels: list[str] = []
        self._pending_save_view_name: Optional[str] = None
        self._pending_selection: Optional[tuple[Optional[str], str]] = None
        self._saved_views_sent = False
        self._saved_views_dirty = False
        self.saved_view_model_file = saved_camera_file(model_path)
        self.saved_view_source_file = saved_camera_file(source_path)
        self.saved_view_write_scope = "model" if self.saved_view_model_file is not None else "source"
        self.saved_view_write_file = (
            self.saved_view_model_file if self.saved_view_model_file is not None else self.saved_view_source_file
        )
        self.model_saved_views: list[CameraInfo] = []
        self.source_saved_views: list[CameraInfo] = []
        super().__init__(mode)
        self.load_from_disk()

    def _selected_saved_view_key(self) -> Optional[tuple[str, str]]:
        if 0 <= self.current_saved_view < len(self.saved_views):
            scope = "model" if self.current_saved_view < len(self.model_saved_views) else "source"
            return scope, self.saved_views[self.current_saved_view].image_name
        return None

    def _refresh_saved_view_list(self, preferred_key: Optional[tuple[Optional[str], str]] = None):
        if preferred_key is None:
            preferred_key = self._selected_saved_view_key()

        self.saved_views = [*self.model_saved_views, *self.source_saved_views]
        name_counts = Counter(view.image_name for view in self.saved_views)
        self.saved_view_labels = []
        for idx, view in enumerate(self.saved_views):
            scope = "model" if idx < len(self.model_saved_views) else "source"
            if name_counts[view.image_name] > 1:
                self.saved_view_labels.append(f"{view.image_name} [{scope}]")
            else:
                self.saved_view_labels.append(view.image_name)

        self.current_saved_view = -1
        if preferred_key is None:
            return

        preferred_scope, preferred_name = preferred_key
        for idx, view in enumerate(self.saved_views):
            if view.image_name != preferred_name:
                continue
            scope = "model" if idx < len(self.model_saved_views) else "source"
            if preferred_scope is None or scope == preferred_scope:
                self.current_saved_view = idx
                break

    def load_from_disk(self, preferred_key: Optional[tuple[Optional[str], str]] = None):
        seen_paths = set()
        self.model_saved_views = []
        self.source_saved_views = []
        for scope, path in (
            ("model", self.saved_view_model_file),
            ("source", self.saved_view_source_file),
        ):
            if path is None:
                continue
            norm_path = os.path.normcase(os.path.abspath(path))
            if norm_path in seen_paths:
                continue
            seen_paths.add(norm_path)
            views = load_saved_cameras(path)
            if scope == "model":
                self.model_saved_views = views
            else:
                self.source_saved_views = views
        self._refresh_saved_view_list(preferred_key=preferred_key)

    def onconnect(self):
        self._saved_views_sent = False
        if self.mode != ViewerMode.CLIENT:
            self.load_from_disk(preferred_key=self._selected_saved_view_key())

    def clear_selection(self):
        self.current_saved_view = -1

    def _capture_current_view(self, name: str) -> CameraInfo:
        return CameraInfo(
            uid=0,
            R=np.asarray(self.camera_widget.to_world[:3, :3], dtype=np.float32),
            T=np.zeros(3, dtype=np.float32),
            origin=np.asarray(self.camera_widget.origin, dtype=np.float32),
            fov_x=float(self.camera_widget.fov_x),
            fov_y=float(self.camera_widget.fov_y),
            image_path="",
            image_name=name,
            image_width=0,
            image_height=0,
            is_test=False,
        )

    def _save_current_view(self, raw_name: str) -> bool:
        name = raw_name.strip()
        if not name:
            self.saved_view_status = "Enter a name before saving a view."
            return False
        if self.saved_view_write_file is None:
            self.saved_view_status = "No model/source path is available for saving views."
            return False

        saved_view = self._capture_current_view(name)
        target_list = (
            self.model_saved_views if self.saved_view_write_scope == "model" else self.source_saved_views
        )
        updated_target_list = upsert_saved_camera(target_list, saved_view)
        if self.saved_view_write_scope == "model":
            self.model_saved_views = updated_target_list
        else:
            self.source_saved_views = updated_target_list

        try:
            write_saved_cameras(self.saved_view_write_file, updated_target_list)
        except OSError as exc:
            self.saved_view_status = f"Failed to save view '{name}': {exc}"
            return False

        self.saved_view_name = name
        self.saved_view_status = f"Saved '{name}' to {os.path.basename(self.saved_view_write_file)}"
        self._refresh_saved_view_list(preferred_key=(self.saved_view_write_scope, saved_view.image_name))
        self._saved_views_dirty = True
        return True

    def _queue_save_current_view(self):
        name = self.saved_view_name.strip()
        if not name:
            self.saved_view_status = "Enter a name before saving a view."
            return

        if self.mode == ViewerMode.CLIENT:
            self._pending_save_view_name = name
            self._pending_selection = (None, name)
            self.saved_view_status = f"Saving view '{name}'..."
        else:
            self._save_current_view(name)

    def _set_saved_views_from_payload(self, model_payload: list[dict], source_payload: list[dict]):
        self.model_saved_views = [CameraInfo.from_json(view_data) for view_data in model_payload]
        self.source_saved_views = [CameraInfo.from_json(view_data) for view_data in source_payload]
        preferred_key = self._pending_selection
        self._refresh_saved_view_list(preferred_key=preferred_key)
        self._pending_selection = None

    def apply_saved_view(self, index: int) -> bool:
        if not (0 <= index < len(self.saved_views)):
            return False

        saved_view = self.saved_views[index]
        self.camera_widget.set_pose(saved_view.R, np.asarray(saved_view.origin, dtype=np.float32))
        self.camera_widget.fov_y = float(saved_view.fov_y)
        self.camera_widget.compute_fov_x()
        self.current_saved_view = index
        return True

    def show_gui(self) -> bool:
        imgui.separator_text("Select Saved View")
        style = imgui.get_style()
        button_width = imgui.calc_text_size("Save").x + 2 * style.frame_padding.x
        available_width = imgui.get_content_region_avail().x
        input_width = max(80.0, (available_width - button_width - style.item_spacing.x) * 0.75)

        imgui.set_next_item_width(input_width)
        _, self.saved_view_name = imgui.input_text_with_hint(
            "##saved_view_name", "Enter a view name", self.saved_view_name
        )
        imgui.same_line()
        if imgui.button("Save"):
            self._queue_save_current_view()

        applied_view = False
        preview_label = (
            self.saved_view_labels[self.current_saved_view]
            if 0 <= self.current_saved_view < len(self.saved_view_labels)
            else ""
        )
        has_saved_views = len(self.saved_views) > 0
        if not has_saved_views:
            imgui.begin_disabled()
        if imgui.begin_combo("##saved_view_picker", preview_label):
            for idx, label in enumerate(self.saved_view_labels):
                is_selected = idx == self.current_saved_view
                clicked, _ = imgui.selectable(label, is_selected)
                if clicked:
                    applied_view = self.apply_saved_view(idx)
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        if not has_saved_views:
            imgui.end_disabled()

        if self.saved_view_status:
            imgui.text_wrapped(self.saved_view_status)
        return applied_view

    def client_send(self):
        if self._pending_save_view_name is None:
            return None, None

        payload = {"save_view_name": self._pending_save_view_name}
        self._pending_save_view_name = None
        return None, payload

    def client_recv(self, _, text):
        if not text:
            return
        if "model_saved_views" in text or "source_saved_views" in text:
            self._set_saved_views_from_payload(
                text.get("model_saved_views", []),
                text.get("source_saved_views", []),
            )
        if "saved_view_status" in text:
            self.saved_view_status = text["saved_view_status"]

    def server_send(self):
        text = {"saved_view_status": self.saved_view_status}
        if not self._saved_views_sent or self._saved_views_dirty:
            text["model_saved_views"] = [view.to_json() for view in self.model_saved_views]
            text["source_saved_views"] = [view.to_json() for view in self.source_saved_views]
            self._saved_views_sent = True
            self._saved_views_dirty = False
        return None, text

    def server_recv(self, _, text):
        if text and "save_view_name" in text:
            self._save_current_view(text["save_view_name"])

    def import_client_modules(self):
        global imgui
        from imgui_bundle import imgui
