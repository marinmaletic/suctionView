"""SAM2 instance segmentation — runs on cpu in a background thread, so it won't freeze the UI."""

import threading

import numpy as np
import open3d.visualization.gui as gui

from settings import SAM2_CHECKPOINT, SAM2_CONFIG

# Loaded on first segmentation call.
_predictor      = None
_predictor_lock = threading.Lock()


def _get_predictor():
    """Load and cache the SAM2 predictor."""
    global _predictor
    if _predictor is not None:
        return _predictor
    with _predictor_lock:
        if _predictor is not None:
            return _predictor
        import os
        import sam2 as _sam2_pkg
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        pkg_file = getattr(_sam2_pkg, "__file__", None)
        pkg_dir  = os.path.dirname(os.path.abspath(pkg_file)) if pkg_file \
                   else os.path.abspath(list(_sam2_pkg.__path__)[0])
        cfg_base = next(
            (d for d in [pkg_dir, os.path.dirname(pkg_dir)]
             if os.path.isdir(os.path.join(d, "configs"))),
            None)
        if cfg_base is None:
            raise RuntimeError("Cannot locate SAM2 'configs/' directory.")
        os.chdir(cfg_base)

        cfg = SAM2_CONFIG
        for prefix in ("/", "sam2/", "./"):
            if cfg.startswith(prefix):
                cfg = cfg[len(prefix):]
        if cfg.startswith("sam2/"):
            cfg = cfg[5:]

        _predictor = SAM2ImagePredictor(
            build_sam2(cfg, SAM2_CHECKPOINT, device="cpu"))
        return _predictor


def segment_sam2(viewer) -> None:
    """Start a background thread that runs SAM2 on the current bounding box."""
    v = viewer
    if v.rgb is None:
        _set_status(v, "Load data first.")
        return
    if v.bb is None:
        _set_status(v, "Draw a bounding box first.")
        return
    _set_status(v, "Starting SAM2...")
    threading.Thread(target=_sam2_thread, args=(v,), daemon=True).start()


def _set_status(viewer, text: str) -> None:
    """Safely update the status label from any thread."""
    gui.Application.instance.post_to_main_thread(
        viewer.win, lambda t=text: setattr(viewer.lbl_seg_status, "text", t))


def _sam2_thread(viewer) -> None:
    """Background thread: runs SAM2 inference and applies the mask."""
    import torch
    v = viewer

    done_flag = threading.Event()

    def _animate() -> None:
        dots = 1
        while not done_flag.wait(timeout=0.5):
            _set_status(v, "Segmenting mask" + "." * dots)
            dots = dots % 3 + 1

    anim_thread = threading.Thread(target=_animate, daemon=True)
    anim_thread.start()

    try:
        x0, y0, x1, y1 = v.bb
        box = np.array([x0, y0, x1, y1], dtype=np.float32)

        predictor = _get_predictor()

        with torch.inference_mode():
            predictor.set_image(v.rgb)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None],
                multimask_output=False,
            )

        mask = masks[np.argmax(scores)].astype(bool)
        n_px = int(mask.sum())

        def _apply() -> None:
            from core.pointcloud import apply_bb, update_plane_range, refresh
            from ui.overlay import update_image_widget
            v.segment_mask        = mask
            v.lbl_seg_status.text = f"Segmented {n_px:,} points"
            apply_bb(v)
            update_plane_range(v)
            v._camera_set = False
            refresh(v, fit=True)
            update_image_widget(v)

        gui.Application.instance.post_to_main_thread(v.win, _apply)

    except Exception as e:
        import traceback
        traceback.print_exc()
        _set_status(v, f"SAM2 error: {e}")
    finally:
        done_flag.set()
        anim_thread.join(timeout=2)


def clear_segmentation(viewer) -> None:
    """Remove the current SAM2 mask and restore the full bounding-box view."""
    v = viewer
    v.segment_mask        = None
    v.lbl_seg_status.text = ""
    v.scene_widget.scene.remove_geometry(v.GEO_GROUND)
    if v._xyz_full is not None:
        from core.pointcloud import apply_bb, update_plane_range, refresh
        from ui.overlay import update_image_widget
        apply_bb(v)
        update_plane_range(v)
        v._camera_set = False
        refresh(v)
        update_image_widget(v)