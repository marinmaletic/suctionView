"""Point cloud processing pipeline.
back-projection + statistical denoising + bounding-box / SAM2 filtering + scene refresh and camera positioning.
"""

import numpy as np
from PIL import Image
import open3d as o3d
import open3d.visualization.rendering as rendering
import open3d.visualization.gui as gui

from settings import FX, FY, CX, CY, DEPTH_SCALE
from utils.colormap import heatmap_colors


def load(viewer) -> None:
    """Load the RGB image and depth array at the current folder index."""
    v = viewer
    try:
        rgb   = np.array(Image.open(v.rgb_path).convert("RGB"))
        depth = np.load(v.dep_path)
        if depth.ndim == 3:
            depth = depth.squeeze()
        if depth.shape != rgb.shape[:2]:
            d     = Image.fromarray(depth.astype(np.float32))
            depth = np.array(d.resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))

        v.rgb   = rgb
        v.depth = depth
        v.img_H = rgb.shape[0]
        v.img_W = rgb.shape[1]
        v.bb    = None
        v.segment_mask  = None
        v._sam2_proc    = None
        v.ground_normal = None
        v.ground_offset = None
        v.lbl_bb.text   = "Box: none"

        backproject_full(v)
        apply_denoise(v)
        apply_bb(v)
        update_plane_range(v)
        v._camera_set = False
        refresh(v)
        from ui.overlay import update_image_widget
        update_image_widget(v)

        d_vals = depth[depth > 0] * DEPTH_SCALE
        v.lbl_stats.text = (
            f"Points: {len(v.xyz):,}\n"
            f"Depth:  {d_vals.min():.2f} - {d_vals.max():.2f} m\n"
            f"Image:  {rgb.shape[1]}x{rgb.shape[0]}")
    except Exception as e:
        v._error(f"Load failed:\n{e}")


def backproject_full(viewer) -> None:
    """Back-project every valid depth pixel into 3-D camera space.

    Uses the pinhole camera model:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth_value * depth_scale
    """
    v   = viewer
    H, W = v.depth.shape
    d    = v.depth.astype(np.float64) * DEPTH_SCALE
    valid = np.isfinite(d) & (d > 0)
    u, pv = np.meshgrid(np.arange(W), np.arange(H))
    uf = u[valid].astype(np.float64)
    vf = pv[valid].astype(np.float64)
    z  = d[valid]
    v._raw_xyz = np.stack([(uf - CX) * z / FX, (vf - CY) * z / FY, z], axis=1)
    v._raw_col = v.rgb[valid].astype(np.float64) / 255.0
    v._raw_u   = u[valid]
    v._raw_v   = pv[valid]


def apply_denoise(viewer) -> None:
    """Remove statistical outliers from the raw cloud.
    The cleaned cloud is stored in ``viewer._xyz_full``.
    """
    v = viewer
    if not v.denoise_enabled or v._raw_xyz is None:
        v._xyz_full = v._raw_xyz.copy()
        v._col_full = v._raw_col.copy()
        v._full_u   = v._raw_u.copy()
        v._full_v   = v._raw_v.copy()
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(v._raw_xyz)
        pcd.colors = o3d.utility.Vector3dVector(v._raw_col)
        pcd, ind   = pcd.remove_statistical_outlier(
            nb_neighbors=v.denoise_nb, std_ratio=v.denoise_std)
        v._xyz_full = np.asarray(pcd.points)
        v._col_full = np.asarray(pcd.colors)
        v._full_u   = v._raw_u[ind]
        v._full_v   = v._raw_v[ind]

    v.ground_normal = None
    v.ground_offset = None
    if v.plane_mode == "ground":
        from core.ground import detect_ground
        detect_ground(v)


def apply_bb(viewer) -> None:
    """Filter the denoised cloud by the current bounding box and SAM2 mask.
    The resulting active cloud is stored in ``viewer.xyz``.
    """
    v = viewer
    if v.bb is None or v._xyz_full is None:
        v.xyz         = v._xyz_full.copy()   if v._xyz_full is not None else None
        v.colors_orig = v._col_full.copy()   if v._col_full is not None else None
        v._cur_u      = v._full_u.copy()     if v._full_u   is not None else None
        v._cur_v      = v._full_v.copy()     if v._full_v   is not None else None
    else:
        u0, v0, u1, v1 = v.bb
        mask = ((v._full_u >= u0) & (v._full_u <= u1) &
                (v._full_v >= v0) & (v._full_v <= v1))
        v.xyz         = v._xyz_full[mask]
        v.colors_orig = v._col_full[mask]
        v._cur_u      = v._full_u[mask]
        v._cur_v      = v._full_v[mask]

    v.suction_scores = None

    if v.segment_mask is not None and v.xyz is not None and v._cur_u is not None:
        keep          = v.segment_mask[v._cur_v, v._cur_u]
        v.xyz         = v.xyz[keep]
        v.colors_orig = v.colors_orig[keep]
        v._cur_u      = v._cur_u[keep]
        v._cur_v      = v._cur_v[keep]

    if v.ground_normal is not None and v.xyz is not None:
        v.heights = (v.xyz @ v.ground_normal) - v.ground_offset
    else:
        v.heights = None


def refresh(viewer, fit: bool | None = None) -> None:
    """Rebuild the 3-D scene geometry from the current viewer state.

    Args:
        fit: If ``True``, reposition the camera to frame the current cloud.
             If ``False``, keep the existing view.  ``None`` uses the
             default logic: fit only if the camera has not been set yet.
    """
    v = viewer
    if v.xyz is None or len(v.xyz) == 0:
        return

    xyz    = v.xyz
    colors = v.colors_orig.copy()

    axis_vals = xyz[:, 2] if (v.plane_mode == "depth" or v.heights is None) else v.heights

    if v.suction_mode is not None and v.suction_scores is not None:
        above_mask = v.suction_scores >= v.suction_thresh
        colors     = heatmap_colors(v.suction_scores)
        colors[~above_mask] = v.colors_orig[~above_mask]

    if v.plane_on:
        in_sl = np.abs(axis_vals - v.plane_height) < v.plane_thickness
        colors[in_sl] = [1.0, 0.60, 0.05]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    mat = rendering.MaterialRecord()
    mat.shader     = "defaultUnlit"
    mat.point_size = v.point_size

    sc = v.scene_widget.scene
    sc.remove_geometry(v.GEO_CLOUD)
    sc.add_geometry(v.GEO_CLOUD, pcd, mat)
    draw_plane(v, xyz)

    # Dimmed background cloud (shown when a SAM2 mask is active).
    sc.remove_geometry(v.GEO_GROUND)
    if v.show_ground and v.segment_mask is not None and v._xyz_full is not None:
        if v._full_u is not None:
            bg_keep = ~v.segment_mask[v._full_v, v._full_u]
            bg_xyz  = v._xyz_full[bg_keep]
            bg_col  = v._col_full[bg_keep] * 0.25
            if len(bg_xyz) > 0:
                bg_pcd = o3d.geometry.PointCloud()
                bg_pcd.points = o3d.utility.Vector3dVector(bg_xyz)
                bg_pcd.colors = o3d.utility.Vector3dVector(np.clip(bg_col, 0.0, 1.0))
                bg_mat = rendering.MaterialRecord()
                bg_mat.shader     = "defaultUnlit"
                bg_mat.point_size = max(1.0, v.point_size * 0.6)
                sc.add_geometry(v.GEO_GROUND, bg_pcd, bg_mat)

    should_fit = fit if fit is not None else (not v._camera_set)
    if should_fit:
        fit_camera(v)
        v._camera_set = True


def draw_plane(viewer, xyz: np.ndarray) -> None:
    """Render the plane slicer grid as a line set."""
    v  = viewer
    sc = v.scene_widget.scene
    sc.remove_geometry(v.GEO_PLANE)
    if not v.plane_on or len(xyz) == 0:
        return

    if v.plane_mode == "depth" or v.ground_normal is None:
        x0, x1 = float(xyz[:, 0].min()), float(xyz[:, 0].max())
        y0, y1 = float(xyz[:, 1].min()), float(xyz[:, 1].max())
        px = (x1 - x0) * 0.05; py = (y1 - y0) * 0.05
        x0 -= px; x1 += px; y0 -= py; y1 += py
        z = v.plane_height
        N = 16; pts, lines = [], []
        for xi in np.linspace(x0, x1, N):
            i = len(pts); pts += [[xi, y0, z], [xi, y1, z]]; lines.append([i, i + 1])
        for yi in np.linspace(y0, y1, N):
            i = len(pts); pts += [[x0, yi, z], [x1, yi, z]]; lines.append([i, i + 1])
    else:
        n      = v.ground_normal
        d      = v.ground_offset
        centre = xyz.mean(axis=0)
        centre = centre - n * (np.dot(n, centre) - d) + n * v.plane_height
        ref    = np.array([0, 1, 0]) if abs(n[1]) < 0.9 else np.array([1, 0, 0])
        t1     = np.cross(n, ref); t1 /= np.linalg.norm(t1)
        t2     = np.cross(n, t1)
        ext    = float(np.ptp(xyz, axis=0).max()) * 0.6
        N = 16; pts, lines = [], []
        for u in np.linspace(-ext, ext, N):
            i = len(pts)
            pts += [(centre + u * t1 - ext * t2).tolist(),
                    (centre + u * t1 + ext * t2).tolist()]
            lines.append([i, i + 1])
        for u in np.linspace(-ext, ext, N):
            i = len(pts)
            pts += [(centre - ext * t1 + u * t2).tolist(),
                    (centre + ext * t1 + u * t2).tolist()]
            lines.append([i, i + 1])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.paint_uniform_color([0.2, 0.65, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader     = "unlitLine"
    mat.line_width = 1.5
    sc.add_geometry(v.GEO_PLANE, ls, mat)


def update_plane_range(viewer) -> None:
    """Sync the plane slicer slider range to the current cloud extent."""
    v = viewer
    if v.xyz is None or len(v.xyz) == 0:
        return
    if v.plane_mode == "depth":
        lo, hi = float(v.xyz[:, 2].min()), float(v.xyz[:, 2].max())
    else:
        if v.heights is None:
            return
        lo, hi = float(v.heights.min()), float(v.heights.max())
    mid = (lo + hi) / 2.0
    v.sld_height.set_limits(lo, hi)
    v.sld_height.double_value = mid
    v.plane_height = mid
    v.lbl_height.text = f"{mid:.4f} m"


def fit_camera(viewer) -> None:
    """Reposition the camera to frame the currently displayed point cloud.
    """
    v = viewer
    if v.xyz is None or len(v.xyz) == 0:
        return
    centre = v.xyz.mean(axis=0).astype(np.float64)
    span   = v.xyz.max(axis=0) - v.xyz.min(axis=0)
    extent = float(span.max())
    if extent < 1e-4:
        extent = 1.0
    eye = centre + np.array([0.0, 0.0, -extent * 1.2])
    v.scene_widget.look_at(centre, eye, np.array([0.0, -1.0, 0.0]))