"""Viewer — application state and window bootstrap.
This module defines the Viewer class, which encapsulates all application state and logic, and constructs the Open3D GUI window. 
"""

import os
import numpy as np


import sys
sys.modules.setdefault("open3d.ml", type(sys)("open3d.ml"))

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from ui.layout  import on_layout
from ui.sidebar import build_sidebar
from ui.overlay import build_toolbar, update_image_widget, on_img_mouse
from core.pointcloud import load, fit_camera


class Viewer:

    # Geometry keys used to add/remove objects from the Open3D scene.
    GEO_CLOUD  = "cloud"
    GEO_PLANE  = "plane"
    GEO_GROUND = "ground_bg"

    def __init__(
        self,
        preload_rgb: str | None    = None,
        preload_depth: str | None  = None,
        preload_folder: str | None = None,
    ):
        # Raw cloud (all valid depth pixels, before bounding box)
        self._raw_xyz  = None
        self._raw_col  = None
        self._raw_u    = None
        self._raw_v    = None
        # Denoised cloud (statistical outlier removal applied)
        self._xyz_full = None
        self._col_full = None
        self._full_u   = None
        self._full_v   = None

        # Active cloud: bounding-box and/or SAM2 mask applied.
        # This is the cloud used for all scoring and display.
        self.xyz         = None
        self.colors_orig = None
        self.heights     = None   # height above ground plane, or None

        # RGB image and depth map for the current frame
        self.rgb    = None
        self.depth  = None
        self.img_H  = 0
        self.img_W  = 0

        # Bounding box drawn by the user on the 2-D image panel.
        # Stored as (u0, v0, u1, v1) in image pixel coordinates.
        self.bb              = None
        self.segment_mask    = None   # H×W boolean mask from SAM2
        self._sam2_proc      = None
        self.show_ground     = True   # render dimmed background behind segmented object
        self._bb_drag_start  = None
        self._bb_drag_cur    = None
        self._img_widget_w   = 1
        self._img_widget_h   = 1
        self._img_win_x      = 0
        self._img_win_y      = 0

        # Ground plane (detected by RANSAC on the full denoised cloud)
        self.ground_normal = None
        self.ground_offset = None

        # Plane slicer
        self.plane_on        = False
        self.plane_height    = 0.0
        self._height_base    = 0.0
        self.plane_thickness = 0.003  # slab half-thickness in metres
        self.plane_mode      = "ground"

        # Statistical outlier removal parameters
        self.denoise_enabled = True
        self.denoise_nb      = 5
        self.denoise_std     = 2.0

        # 3-D display
        self._camera_set = False
        self.point_size  = 3.0

        # Suction quality scoring
        self.suction_mode   = None   # "knn" | "sobel" | "ransac" | None
        self.suction_scores = None   # (N,) float32 per-point scores
        self.suction_k      = 30     # KNN neighbour count for normal estimation
        self.suction_win    = 25     # std-filter window size in pixels
        self.cup_size_mm    = None   # None = manual window; 30 or 50 = preset
        self.suction_thresh = 0.5    # minimum score to show in colour
        self.ransac_iters   = 50     # plane hypotheses tested per point
        self.ransac_tol_mm  = 5      # inlier distance tolerance in mm

        # Folder browser
        self._folder_pairs   = []
        self._folder_idx     = 0
        self._preload_folder = preload_folder
        self._preload_rgb    = preload_rgb
        self._preload_depth  = preload_depth

        self._build_window()

    # ── Window construction ───────────────────────────────────────────────────

    def _build_window(self) -> None:
        app = gui.Application.instance
        app.initialize()
        self.em = (getattr(app, "font_size", None)
                   or getattr(getattr(app, "theme", None), "font_size", None)
                   or 14)

        self.win = app.create_window("RGB-D Viewer", 1600, 960)

        # 3-D scene widget
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.win.renderer)
        self.scene_widget.scene.set_background([0.10, 0.10, 0.11, 1.0])
        self.scene_widget.scene.scene.enable_sun_light(False)
        try:
            self.scene_widget.scene.scene.enable_indirect_light(True)
            self.scene_widget.scene.scene.set_indirect_light_intensity(25000)
        except Exception:
            pass

        # 2-D image panel
        blank = o3d.geometry.Image(np.zeros((3, 4, 3), dtype=np.uint8))
        self.img_widget = gui.ImageWidget(blank)
        self.img_widget.set_on_mouse(lambda e: on_img_mouse(self, e))

        toolbar  = build_toolbar(self)
        sidebar  = build_sidebar(self)

        self.win.add_child(self.scene_widget)
        self.win.add_child(self.img_widget)
        self.win.add_child(toolbar)
        self.win.add_child(sidebar)
        self.win.set_on_layout(lambda ctx: on_layout(self, ctx))

        if self._preload_folder:
            gui.Application.instance.post_to_main_thread(
                self.win, lambda: self._open_folder(self._preload_folder))
        elif self._preload_rgb and self._preload_depth:
            self._folder_pairs = [(self._preload_rgb, self._preload_depth)]
            self._folder_idx   = 0
            self._update_nav_labels()
            gui.Application.instance.post_to_main_thread(self.win, self._load_current)

    # Folder / file navigation

    def _pick_folder(self) -> None:
        dlg = gui.FileDialog(gui.FileDialog.OPEN_DIR, "Select image folder", self.win.theme)
        dlg.set_on_cancel(self.win.close_dialog)
        dlg.set_on_done(self._on_folder_done)
        self.win.show_dialog(dlg)

    def _on_folder_done(self, path: str) -> None:
        self.win.close_dialog()
        self._open_folder(path)

    def _open_folder(self, folder: str) -> None:
        """Scan a folder for RGB/depth pairs (image.png + image.npy)."""
        exts  = (".png", ".jpg", ".jpeg")
        pairs = []
        try:
            for fname in sorted(os.listdir(folder)):
                if not fname.lower().endswith(exts):
                    continue
                stem  = os.path.splitext(fname)[0]
                depth = os.path.join(folder, stem + ".npy")
                if os.path.isfile(depth):
                    pairs.append((os.path.join(folder, fname), depth))
        except Exception as e:
            self._error(f"Could not read folder:\n{e}")
            return

        if not pairs:
            self._error("No matching RGB + .npy depth pairs found in folder.")
            return

        self._folder_pairs = pairs
        self._folder_idx   = 0
        self.lbl_folder.text = os.path.basename(folder)
        self._update_nav_labels()
        self._load_current()

    def _update_nav_labels(self) -> None:
        total = len(self._folder_pairs)
        idx   = self._folder_idx
        self.lbl_img_idx.text  = f"{idx + 1} / {total}"
        self.btn_prev.enabled  = idx > 0
        self.btn_next.enabled  = idx < total - 1

    def _prev_image(self) -> None:
        if self._folder_idx > 0:
            self._folder_idx -= 1
            self._update_nav_labels()
            self._load_current()

    def _next_image(self) -> None:
        if self._folder_idx < len(self._folder_pairs) - 1:
            self._folder_idx += 1
            self._update_nav_labels()
            self._load_current()

    def _load_current(self) -> None:
        if not self._folder_pairs:
            return
        self.rgb_path, self.dep_path = self._folder_pairs[self._folder_idx]
        load(self)
        gui.Application.instance.post_to_main_thread(
            self.win, lambda: update_image_widget(self))
        self.win.post_redraw()

    # SAM2 / bounding-box wrappers

    def _segment_sam2(self) -> None:
        from core.segmentation import segment_sam2
        segment_sam2(self)

    def _clear_segmentation(self) -> None:
        from core.segmentation import clear_segmentation
        clear_segmentation(self)

    def _clear_bb(self) -> None:
        self.bb              = None
        self._bb_drag_start  = None
        self._bb_drag_cur    = None
        self.lbl_bb.text     = "Box: none"
        if self._xyz_full is not None:
            from core.pointcloud import apply_bb, update_plane_range, refresh
            apply_bb(self)
            update_plane_range(self)
            self._camera_set = False
            refresh(self)
            update_image_widget(self)

    # ── Utility ───────────────────────────────────────────────────────────────

    def _error(self, msg: str) -> None:
        em  = self.em
        dlg = gui.Dialog("Error")
        body = gui.Vert(em, gui.Margins(em, em, em, em))
        body.add_child(gui.Label(msg))
        ok = gui.Button("OK")
        ok.set_on_clicked(self.win.close_dialog)
        body.add_child(ok)
        dlg.add_child(body)
        self.win.show_dialog(dlg)

    def _refresh(self, fit: bool | None = None) -> None:
        """Convenience wrapper so modules can call viewer._refresh()."""
        from core.pointcloud import refresh
        refresh(self, fit=fit)

    def run(self) -> None:
        gui.Application.instance.run()