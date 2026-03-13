"""Project settings loader.
Reads config.yaml and the camera intrinsics JSON file it points to
"""

import json
import os

# ── Locate project root ───────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_yaml(path: str) -> dict:

    result = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            # Remove inline comments
            if " #" in val:
                val = val[:val.index(" #")].strip()
            # Type coercion
            if val.replace(".", "", 1).lstrip("-").isdigit():
                val = float(val) if "." in val else int(val)
            result[key] = val
    return result


# Load config.yaml
_cfg_path = os.path.join(_ROOT, "config.yaml")
if not os.path.exists(_cfg_path):
    raise FileNotFoundError(
        f"config.yaml not found at {_cfg_path}\n"
        "Copy config.yaml to the project root and edit as needed.")

_cfg = _load_yaml(_cfg_path)

# Load intrinsics.json
_intr_rel  = str(_cfg.get("intrinsics_file", "intrinsics.json"))
_intr_path = _intr_rel if os.path.isabs(_intr_rel) \
             else os.path.join(_ROOT, _intr_rel)

if not os.path.exists(_intr_path):
    raise FileNotFoundError(
        f"Camera intrinsics file not found: {_intr_path}\n"
        f"Set 'intrinsics_file' in config.yaml to the correct path.\n"
        f"Expected JSON with keys: fx, fy, ppx, ppy (and optionally depth_scale).")

with open(_intr_path) as _f:
    _intr = json.load(_f)

# Camera intrinsics 
FX: float = float(_intr["fx"])
FY: float = float(_intr["fy"])
CX: float = float(_intr["ppx"])   # principal point x
CY: float = float(_intr["ppy"])   # principal point y


DEPTH_SCALE: float = float(
    _intr.get("depth_scale", _cfg.get("depth_scale", 0.001)))

# SAM2 paths
SAM2_CHECKPOINT: str = str(_cfg.get(
    "sam2_checkpoint", "/root/sam2/checkpoints/sam2.1_hiera_small.pt"))
SAM2_CONFIG: str = str(_cfg.get(
    "sam2_config", "sam2/configs/sam2.1/sam2.1_hiera_s.yaml"))

# Point cloud processing
DENOISE_NB:  int   = int(_cfg.get("denoise_nb_neighbors", 20))
DENOISE_STD: float = float(_cfg.get("denoise_std_ratio",  2.0))

# Suction scoring defaults
DEFAULT_KNN_K:         int   = int(_cfg.get("default_knn_k",         30))
DEFAULT_STD_WIN:       int   = int(_cfg.get("default_std_win",        25))
DEFAULT_RANSAC_ITERS:  int   = int(_cfg.get("default_ransac_iters",   50))
DEFAULT_RANSAC_TOL_MM: float = float(_cfg.get("default_ransac_tol_mm", 3))