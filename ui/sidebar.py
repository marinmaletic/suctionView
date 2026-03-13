"""Sidebar panel construction and all widget callbacks."""

import open3d.visualization.gui as gui

from core.pointcloud import (
    apply_denoise, apply_bb, update_plane_range, refresh, fit_camera
)
from core.ground import detect_ground
from suction.compute import compute_suction


# ── Helpers 

def _section(parent: gui.Vert, title: str) -> None:
    lbl = gui.Label(f"  {title}")
    lbl.text_color = gui.Color(0.88, 0.88, 0.88)
    parent.add_child(lbl)


def _lbl(text: str) -> gui.Label:
    l = gui.Label(text)
    l.text_color = gui.Color(0.55, 0.55, 0.55)
    return l


def _divider(parent: gui.Vert) -> None:
    parent.add_child(gui.Label(""))


# ── Section builders 

def _build_files(v, parent: gui.Vert) -> None:
    _section(parent, "FILES")
    btn = gui.Button("Select Folder")
    btn.set_on_clicked(v._pick_folder)
    parent.add_child(btn)
    v.lbl_folder = gui.Label("No folder selected")
    v.lbl_folder.text_color = gui.Color(0.45, 0.45, 0.45)
    parent.add_child(v.lbl_folder)
    # Invisible label kept so internal code that writes lbl_bb.text still works.
    v.lbl_bb         = gui.Label("")
    v.lbl_bb.visible = False
    _divider(parent)


def _build_plane(v, parent: gui.Vert) -> None:
    _section(parent, "PLANE SLICER")
    v.chk_mode_depth = gui.Checkbox("Depth from camera")
    v.chk_mode_depth.checked = False
    v.chk_mode_depth.set_on_checked(lambda c: _on_mode_depth(v, c))
    parent.add_child(v.chk_mode_depth)
    v.chk_mode_ground = gui.Checkbox("Height above ground")
    v.chk_mode_ground.checked = True
    v.chk_mode_ground.set_on_checked(lambda c: _on_mode_ground(v, c))
    parent.add_child(v.chk_mode_ground)
    v.chk_plane = gui.Checkbox("Enable plane")
    v.chk_plane.set_on_checked(lambda c: _on_toggle_plane(v, c))
    parent.add_child(v.chk_plane)
    v.lbl_sld_coarse = _lbl("Height above ground (m)")
    parent.add_child(v.lbl_sld_coarse)
    v.sld_height = gui.Slider(gui.Slider.DOUBLE)
    v.sld_height.set_limits(0.0, 5.0)
    v.sld_height.double_value = 0.0
    v.sld_height.set_on_value_changed(lambda val: _on_height(v, val))
    parent.add_child(v.sld_height)
    parent.add_child(_lbl("Plane thickness (mm)"))
    v.sld_thick = gui.Slider(gui.Slider.INT)
    v.sld_thick.set_limits(1, 10)
    v.sld_thick.int_value = 3
    v.sld_thick.set_on_value_changed(lambda val: _on_thickness(v, val))
    parent.add_child(v.sld_thick)
    v.lbl_height = gui.Label("0.0000 m")
    parent.add_child(v.lbl_height)
    _divider(parent)


def _build_display(v, parent: gui.Vert) -> None:
    _section(parent, "DISPLAY")
    parent.add_child(_lbl("Point size"))
    v.sld_ptsize = gui.Slider(gui.Slider.DOUBLE)
    v.sld_ptsize.set_limits(0.5, 8.0)
    v.sld_ptsize.double_value = 3.0
    v.sld_ptsize.set_on_value_changed(lambda val: _on_ptsize(v, val))
    parent.add_child(v.sld_ptsize)
    v.chk_show_ground = gui.Checkbox("Show background/ground")
    v.chk_show_ground.checked = True
    v.chk_show_ground.set_on_checked(lambda c: _on_show_ground(v, c))
    parent.add_child(v.chk_show_ground)
    _divider(parent)


def _build_suction(v, parent: gui.Vert) -> None:
    _section(parent, "SUCTION QUALITY")
    parent.add_child(_lbl("Score: 1=good, 0=bad for suction"))

    v.chk_suc_knn = gui.Checkbox("KNN")
    v.chk_suc_knn.checked = False
    v.chk_suc_knn.set_on_checked(lambda c: _on_suc_knn(v, c))
    parent.add_child(v.chk_suc_knn)

    v.chk_suc_sobel = gui.Checkbox("Sobel")
    v.chk_suc_sobel.checked = False
    v.chk_suc_sobel.set_on_checked(lambda c: _on_suc_sobel(v, c))
    parent.add_child(v.chk_suc_sobel)

    v.chk_suc_ransac = gui.Checkbox("RANSAC")
    v.chk_suc_ransac.checked = False
    v.chk_suc_ransac.set_on_checked(lambda c: _on_suc_ransac(v, c))
    parent.add_child(v.chk_suc_ransac)

    parent.add_child(_lbl("RANSAC iterations"))
    v.sld_ransac_iters = gui.Slider(gui.Slider.INT)
    v.sld_ransac_iters.set_limits(10, 200)
    v.sld_ransac_iters.int_value = 50
    v.sld_ransac_iters.set_on_value_changed(
        lambda val: setattr(v, "ransac_iters", int(val)))
    parent.add_child(v.sld_ransac_iters)

    parent.add_child(_lbl("Inlier tolerance (mm)"))
    v.sld_ransac_tol = gui.Slider(gui.Slider.INT)
    v.sld_ransac_tol.set_limits(1, 10)
    v.sld_ransac_tol.int_value = 3
    v.sld_ransac_tol.set_on_value_changed(
        lambda val: setattr(v, "ransac_tol_mm", int(val)))
    parent.add_child(v.sld_ransac_tol)

    parent.add_child(_lbl("Suction cup diameter"))
    row_cup = gui.Horiz(int(v.em * 0.4))
    v.chk_cup_30 = gui.Checkbox("30 mm")
    v.chk_cup_30.checked = False
    v.chk_cup_30.set_on_checked(lambda c: _on_cup_30(v, c))
    row_cup.add_child(v.chk_cup_30)
    v.chk_cup_50 = gui.Checkbox("50 mm")
    v.chk_cup_50.checked = False
    v.chk_cup_50.set_on_checked(lambda c: _on_cup_50(v, c))
    row_cup.add_child(v.chk_cup_50)
    v.chk_cup_manual = gui.Checkbox("manual")
    v.chk_cup_manual.checked = True
    v.chk_cup_manual.set_on_checked(lambda c: _on_cup_manual(v, c))
    row_cup.add_child(v.chk_cup_manual)
    parent.add_child(row_cup)
    v.lbl_cup_info = gui.Label("Window: manual")
    v.lbl_cup_info.text_color = gui.Color(0.45, 0.45, 0.45)
    parent.add_child(v.lbl_cup_info)

    parent.add_child(_lbl("KNN neighbors"))
    v.sld_suc_k = gui.Slider(gui.Slider.INT)
    v.sld_suc_k.set_limits(10, 100)
    v.sld_suc_k.int_value = 30
    v.sld_suc_k.set_on_value_changed(lambda val: setattr(v, "suction_k", int(val)))
    parent.add_child(v.sld_suc_k)

    parent.add_child(_lbl("Std window (N×N pixels)"))
    v.sld_suc_win = gui.Slider(gui.Slider.INT)
    v.sld_suc_win.set_limits(3, 101)
    v.sld_suc_win.int_value = 25
    v.sld_suc_win.set_on_value_changed(lambda val: _on_suc_win_manual(v, val))
    parent.add_child(v.sld_suc_win)

    btn_compute = gui.Button("Compute Suction Scores")
    btn_compute.set_on_clicked(lambda: compute_suction(v))
    btn_compute.background_color = gui.Color(0.18, 0.45, 0.22, 1.0)
    parent.add_child(btn_compute)

    btn_clear = gui.Button("Clear")
    btn_clear.set_on_clicked(lambda: _clear_suction(v))
    parent.add_child(btn_clear)

    parent.add_child(_lbl("Score threshold"))
    v.sld_suc_thresh = gui.Slider(gui.Slider.DOUBLE)
    v.sld_suc_thresh.set_limits(0.0, 1.0)
    v.sld_suc_thresh.double_value = 0.5
    v.sld_suc_thresh.set_on_value_changed(lambda val: _on_suc_thresh(v, val))
    parent.add_child(v.sld_suc_thresh)

    v.lbl_suc_status = gui.Label("Not computed.")
    v.lbl_suc_status.text_color = gui.Color(0.5, 0.5, 0.5)
    parent.add_child(v.lbl_suc_status)
    _divider(parent)


def _build_info(v, parent: gui.Vert) -> None:
    _section(parent, "INFO")
    v.lbl_stats = gui.Label("No data loaded.")
    v.lbl_stats.text_color = gui.Color(0.5, 0.5, 0.5)
    parent.add_child(v.lbl_stats)
    parent.add_stretch()
    btn_reset = gui.Button("Reset Camera")
    btn_reset.set_on_clicked(lambda: fit_camera(v))
    parent.add_child(btn_reset)


def build_sidebar(viewer) -> gui.Vert:
    """Build the complete sidebar and return it. """
    v  = viewer
    m  = gui.Margins(int(v.em * 0.8), int(v.em * 0.5),
                     int(v.em * 0.8), int(v.em * 0.5))
    sb = gui.Vert(int(v.em * 0.45), m)

    _build_files(v, sb)
    _build_plane(v, sb)
    _build_display(v, sb)
    _build_suction(v, sb)
    _build_info(v, sb)

    v.sidebar = sb
    return sb


# ── Callbacks 

def _on_toggle_plane(v, checked: bool) -> None:
    v.plane_on = checked
    refresh(v, fit=False)


def _on_mode_depth(v, checked: bool) -> None:
    if not checked:
        v.chk_mode_depth.checked = True
        return
    v.plane_mode = "depth"
    v.chk_mode_ground.checked = False
    v.lbl_sld_coarse.text = "Depth (m)"
    update_plane_range(v)
    if v.plane_on:
        refresh(v, fit=False)


def _on_mode_ground(v, checked: bool) -> None:
    if not checked:
        v.chk_mode_ground.checked = True
        return
    v.plane_mode = "ground"
    v.chk_mode_depth.checked = False
    v.lbl_sld_coarse.text = "Height above ground (m)"
    if v._xyz_full is not None and v.ground_normal is None:
        detect_ground(v)
        if v.xyz is not None and v.ground_normal is not None:
            import numpy as np
            v.heights = (v.xyz @ v.ground_normal) - v.ground_offset
    update_plane_range(v)
    if v.plane_on:
        refresh(v, fit=False)


def _on_height(v, val: float) -> None:
    v.plane_height = val
    v.lbl_height.text = f"{val:.4f} m"
    if v.plane_on:
        refresh(v, fit=False)


def _on_thickness(v, val: float) -> None:
    v.plane_thickness = int(val) / 1000.0
    if v.plane_on:
        refresh(v, fit=False)


def _on_ptsize(v, val: float) -> None:
    v.point_size = val
    refresh(v, fit=False)


def _on_show_ground(v, checked: bool) -> None:
    v.show_ground = checked
    v.scene_widget.scene.remove_geometry(v.GEO_GROUND)
    refresh(v, fit=False)


def _on_suc_knn(v, checked: bool) -> None:
    if checked:
        v.chk_suc_sobel.checked  = False
        v.chk_suc_ransac.checked = False
        v.suction_mode = "knn"
    else:
        v.suction_mode = None
    v.suction_scores = None
    refresh(v, fit=False)


def _on_suc_sobel(v, checked: bool) -> None:
    if checked:
        v.chk_suc_knn.checked    = False
        v.chk_suc_ransac.checked = False
        v.suction_mode = "sobel"
    else:
        v.suction_mode = None
    v.suction_scores = None
    refresh(v, fit=False)


def _on_suc_ransac(v, checked: bool) -> None:
    if checked:
        v.chk_suc_knn.checked   = False
        v.chk_suc_sobel.checked = False
        v.suction_mode = "ransac"
    else:
        v.suction_mode = None
    v.suction_scores = None
    refresh(v, fit=False)


def _clear_suction(v) -> None:
    v.suction_mode   = None
    v.suction_scores = None
    v.chk_suc_knn.checked    = False
    v.chk_suc_sobel.checked  = False
    v.chk_suc_ransac.checked = False
    v.lbl_suc_status.text    = "Cleared."
    refresh(v, fit=False)


def _on_suc_thresh(v, val: float) -> None:
    v.suction_thresh = val
    if v.suction_scores is not None:
        lo = float(v.suction_scores.min())
        hi = float(v.suction_scores.max())
        v.lbl_suc_status.text = f"Range: {lo:.3f} - {hi:.3f}"
        refresh(v, fit=False)
        from ui.overlay import update_suction_image
        update_suction_image(v)


def _on_cup_30(v, checked: bool) -> None:
    if checked:
        v.chk_cup_50.checked     = False
        v.chk_cup_manual.checked = False
        v.cup_size_mm = 30
        from suction.compute import _apply_cup_size
        _apply_cup_size(v)
    else:
        v.cup_size_mm = None
        v.lbl_cup_info.text = "Window: manual"


def _on_cup_50(v, checked: bool) -> None:
    if checked:
        v.chk_cup_30.checked     = False
        v.chk_cup_manual.checked = False
        v.cup_size_mm = 50
        from suction.compute import _apply_cup_size
        _apply_cup_size(v)
    else:
        v.cup_size_mm = None
        v.lbl_cup_info.text = "Window: manual"


def _on_cup_manual(v, checked: bool) -> None:
    if checked:
        v.chk_cup_30.checked = False
        v.chk_cup_50.checked = False
        v.cup_size_mm = None
        v.lbl_cup_info.text = "Window: manual"
    else:
        v.chk_cup_manual.checked = True 


def _on_suc_win_manual(v, val: float) -> None:
    v.suction_win = int(val)
    if v.cup_size_mm is not None:
        v.cup_size_mm = None
        v.chk_cup_30.checked     = False
        v.chk_cup_50.checked     = False
        v.chk_cup_manual.checked = True
        v.lbl_cup_info.text      = "Window: manual"