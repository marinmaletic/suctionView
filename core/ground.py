"""Ground plane detection using Open3D RANSAC plane segmentation."""

import numpy as np
import open3d as o3d


def detect_ground(viewer) -> None:
    
    v = viewer
    if v._xyz_full is None or len(v._xyz_full) < 100:
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v._xyz_full)

    plane_model, _ = pcd.segment_plane(
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000,
    )
    a, b, c, d = plane_model
    norm = np.linalg.norm([a, b, c])
    n    = np.array([a, b, c]) / norm

    if np.dot(n, np.array([0.0, 1.0, 0.0])) < 0:
        n = -n
        d = -d

    v.ground_normal = n
    v.ground_offset = -d / norm
    print(f"[ground] normal={n.round(3)}")
