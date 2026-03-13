"""2-D image overlay: toolbar row, bounding-box drawing, and score heatmap."""

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui

from utils.colormap import heatmap_colors


def build_toolbar(viewer) -> gui.Horiz:
    """Build the single toolbar row that floats above the 2-D image panel.    """
    v  = viewer

    def _sep():
        arr = np.full((32, 3, 3), 70, dtype=np.uint8)
        return gui.ImageWidget(o3d.geometry.Image(arr))

    rm = gui.Margins(6, 0, 6, 0)
    row = gui.Horiz(5, rm)

    # Image navigation
    v.btn_prev = gui.Button(" < ")
    v.btn_prev.set_on_clicked(v._prev_image)
    v.btn_prev.enabled = False
    v.btn_next = gui.Button(" > ")
    v.btn_next.set_on_clicked(v._next_image)
    v.btn_next.enabled = False
    wrap = gui.Vert(0)
    wrap.add_stretch()
    v.lbl_img_idx = gui.Label("0 / 0")
    v.lbl_img_idx.text_color = gui.Color(0.75, 0.75, 0.75)
    wrap.add_child(v.lbl_img_idx)
    wrap.add_stretch()
    row.add_child(v.btn_prev)
    row.add_child(wrap)
    row.add_child(v.btn_next)

    row.add_child(_sep())

    btn_clear_bb = gui.Button("Clear Box")
    btn_clear_bb.set_on_clicked(v._clear_bb)
    row.add_child(btn_clear_bb)

    row.add_child(_sep())

    btn_seg  = gui.Button("Segment Mask")
    btn_clr  = gui.Button("Clear Mask")
    btn_seg.set_on_clicked(v._segment_sam2)
    btn_clr.set_on_clicked(v._clear_segmentation)
    row.add_child(btn_seg)
    row.add_child(btn_clr)

    row.add_child(_sep())

    v.lbl_seg_status = gui.Label("")
    v.lbl_seg_status.text_color = gui.Color(0.55, 0.85, 0.55)
    row.add_child(v.lbl_seg_status)
    row.add_stretch()  

    v.row_nav  = row
    v.row_bb   = row 
    v.row_sam2 = row
    return row


def update_image_widget(viewer, preview_bb=None) -> None:
    """Redraw the 2-D image panel with bounding-box overlays. """
    
    v = viewer
    if v.rgb is None:
        return
    img = v.rgb.copy()

    def _draw_rect(im, bb, color):
        u0, pv0, u1, pv1 = bb
        u0  = max(0, u0);  pv0 = max(0, pv0)
        u1  = min(v.img_W - 1, u1); pv1 = min(v.img_H - 1, pv1)
        im[pv0:pv0 + 2,    u0:u1 + 1]  = color
        im[pv1 - 1:pv1 + 1, u0:u1 + 1] = color
        im[pv0:pv1 + 1,    u0:u0 + 2]  = color
        im[pv0:pv1 + 1,    u1 - 1:u1 + 1] = color

    if v.bb:
        _draw_rect(img, v.bb, [255, 255, 255])
    if preview_bb:
        _draw_rect(img, preview_bb, [255, 220, 0])

    v.img_widget.update_image(o3d.geometry.Image(np.ascontiguousarray(img)))


def update_suction_image(viewer) -> None:
    """Overlay the suction score heatmap on the 2-D image panel."""
    v = viewer
    if v.rgb is None or v.suction_scores is None:
        return
    H, W = v.img_H, v.img_W
    u, pv = v._cur_u, v._cur_v

    score_map = np.zeros((H, W), dtype=np.float32)
    raw       = np.where(v.suction_scores >= v.suction_thresh, v.suction_scores, 0.0)
    score_map[pv, u] = raw

    heat_rgb  = (heatmap_colors(score_map.reshape(-1)).reshape(H, W, 3) * 255).astype(np.uint8)
    has_score = (score_map > 0)[..., None]
    blended   = np.where(
        has_score,
        v.rgb.astype(np.float32) * 0.7 + heat_rgb.astype(np.float32) * 0.3,
        v.rgb.astype(np.float32),
    ).astype(np.uint8)

    if v.bb:
        u0, pv0, u1, pv1 = v.bb
        u0 = max(0, u0); pv0 = max(0, pv0)
        u1 = min(W - 1, u1); pv1 = min(H - 1, pv1)
        blended[pv0:pv0 + 2,     u0:u1 + 1]  = [255, 255, 255]
        blended[pv1 - 1:pv1 + 1, u0:u1 + 1]  = [255, 255, 255]
        blended[pv0:pv1 + 1,     u0:u0 + 2]  = [255, 255, 255]
        blended[pv0:pv1 + 1,     u1 - 1:u1 + 1] = [255, 255, 255]

    v.img_widget.update_image(o3d.geometry.Image(np.ascontiguousarray(blended)))


def widget_to_img(viewer, abs_x: int, abs_y: int) -> tuple[int, int]:
    """Convert absolute window mouse coordinates to image pixel coordinates."""
    v = viewer
    if v.img_H == 0:
        return 0, 0
    wx    = abs_x - getattr(v, "_img_win_x", 0)
    wy    = abs_y - getattr(v, "_img_win_y", 0)
    scale = min(v._img_widget_w / v.img_W, v._img_widget_h / v.img_H)
    ox    = (v._img_widget_w - v.img_W * scale) / 2
    oy    = (v._img_widget_h - v.img_H * scale) / 2
    ix    = int((wx - ox) / scale)
    iy    = int((wy - oy) / scale)
    return (max(0, min(v.img_W - 1, ix)),
            max(0, min(v.img_H - 1, iy)))


def get_drag_bb(viewer):
    """Return the bounding box currently being drawn, or ``None``."""
    v = viewer
    if not v._bb_drag_start or not v._bb_drag_cur:
        return None
    x0, y0 = v._bb_drag_start
    x1, y1 = v._bb_drag_cur
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def on_img_mouse(viewer, event) -> gui.Widget.EventCallbackResult:
    """Handle mouse events on the 2-D image panel.

    Two-click bounding box:
    - First click sets the first corner (yellow preview follows the cursor).
    - Second click confirms the box and filters the 3-D view.
    """
    v = viewer
    if v.rgb is None:
        return gui.Widget.EventCallbackResult.IGNORED

    if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
        pt = widget_to_img(v, event.x, event.y)
        if v._bb_drag_start is None:
            v._bb_drag_start = pt
            v._bb_drag_cur   = pt
            update_image_widget(v, preview_bb=None)
        else:
            v._bb_drag_cur = pt
            bb = get_drag_bb(v)
            v._bb_drag_start = None
            v._bb_drag_cur   = None
            if bb and (bb[2] - bb[0]) > 5 and (bb[3] - bb[1]) > 5:
                v.bb = bb
                u0, pv0, u1, pv1 = bb
                v.lbl_bb.text = f"Box: ({u0},{pv0})-({u1},{pv1})"
                from core.pointcloud import apply_bb, update_plane_range, refresh
                apply_bb(v)
                update_plane_range(v)
                v._camera_set = False
                refresh(v)
            update_image_widget(v)
        return gui.Widget.EventCallbackResult.HANDLED

    elif event.type == gui.MouseEvent.Type.MOVE:
        if v._bb_drag_start is not None:
            v._bb_drag_cur = widget_to_img(v, event.x, event.y)
            update_image_widget(v, preview_bb=get_drag_bb(v))
        return gui.Widget.EventCallbackResult.HANDLED

    return gui.Widget.EventCallbackResult.IGNORED