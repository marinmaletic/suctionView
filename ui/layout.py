"""Window layout callback.

Called by Open3D whenever the window is resized.  Positions the 3-D scene,
2-D image overlay, toolbar row, and sidebar relative to the current window
content rect.
"""

import open3d.visualization.gui as gui


def on_layout(viewer, ctx) -> None:
    """Compute and assign frame rects for all top-level widgets."""
    v  = viewer
    r  = v.win.content_rect
    sw = int(v.em * 19)      # sidebar width in pixels
    vw = r.width - sw        # remaining width for 3-D view

    # Image overlay: upper-right corner of the 3-D view, aspect-preserved.
    if v.img_H > 0 and v.img_W > 0:
        iw = int(vw * 0.35)
        ih = int(iw * v.img_H / v.img_W)
        ih = min(ih, int(r.height * 0.40))
        iw = int(ih * v.img_W / v.img_H)
    else:
        iw = int(vw * 0.35)
        ih = int(r.height * 0.25)

    v.scene_widget.frame = gui.Rect(r.x + sw, r.y, vw, r.height)

    ix = r.x + sw + vw - iw - 4   # right-aligned with 4 px margin
    rh = int(v.em * 2.6)           # toolbar row height
    v.row_nav.frame   = gui.Rect(ix, r.y + 4, iw, rh)
    iy = r.y + 4 + rh + 4
    v.img_widget.frame = gui.Rect(ix, iy, iw, ih)
    v.sidebar.frame    = gui.Rect(r.x, r.y, sw, r.height)

    # Store image widget position for mouse-to-pixel coordinate conversion.
    v._img_win_x    = ix
    v._img_win_y    = iy
    v._img_widget_w = iw
    v._img_widget_h = ih
