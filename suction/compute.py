"""Suction score computation dispatcher.

Reads the current viewer state, calls the appropriate scoring module,
and writes the results back to the viewer.
"""

import traceback
import numpy as np

from suction.knn   import score_knn
from suction.sobel import score_sobel
from suction.ransac import score_ransac
from settings import FX


def compute_suction(viewer) -> None:
    """Compute suction scores for the currently displayed point cloud.
    The scores are written back to the viewer, and the status label is updated
    """
    v = viewer
    if v.xyz is None or len(v.xyz) == 0 or v.suction_mode is None:
        v.lbl_suc_status.text = "Load data and select a mode first."
        return

    if v.cup_size_mm is not None:
        _apply_cup_size(v)

    v.lbl_suc_status.text = "Computing..."

    xyz      = v.xyz
    u, pv    = v._cur_u, v._cur_v
    H, W     = v.img_H, v.img_W
    win      = v.suction_win | 1
    approach = _approach_direction(v)

    try:
        if v.suction_mode == "knn":
            v.suction_scores = score_knn(xyz, u, pv, H, W, v.suction_k, win, approach)

        elif v.suction_mode == "sobel":
            v.suction_scores = score_sobel(xyz, u, pv, H, W, win, approach)

        elif v.suction_mode == "ransac":
            v.suction_scores = score_ransac(
                xyz, u, pv, H, W, win,
                v.cup_size_mm,
                v.ransac_iters,
                v.ransac_tol_mm,
                approach,
                FX,
            )

        lo = float(v.suction_scores.min())
        hi = float(v.suction_scores.max())
        v.lbl_suc_status.text = f"Range: {lo:.3f} - {hi:.3f}"
        v._refresh(fit=False)
        from ui.overlay import update_suction_image
        update_suction_image(v)

    except Exception as e:
        v.lbl_suc_status.text = f"Error: {e}"
        traceback.print_exc()


def _approach_direction(viewer) -> np.ndarray:
    """Return the robot gripper approach direction as a unit vector.    """
    v = viewer
    if v.plane_mode == "ground" and v.ground_normal is not None:
        return (-v.ground_normal).astype(np.float32)
    return np.array([0.0, 0.0, -1.0], dtype=np.float32)


def _apply_cup_size(viewer) -> None:
    """Recalculate the std window in pixels to match the active cup preset
    at the current median scene depth, then update the slider label."""
    v = viewer
    if v.cup_size_mm is None or v.xyz is None or len(v.xyz) == 0:
        return
    median_depth = float(np.median(v.xyz[:, 2]))
    if median_depth <= 0:
        return
    win_px = int(round((v.cup_size_mm / 1000.0) * FX / median_depth))
    win_px = max(3, win_px | 1)
    win_px = min(101, win_px)
    v.suction_win          = win_px
    v.sld_suc_win.int_value = win_px
    v.lbl_cup_info.text    = (
        f"{v.cup_size_mm}mm @ {median_depth:.2f}m = {win_px}px window")