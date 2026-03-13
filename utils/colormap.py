import numpy as np


def heatmap_colors(t: np.ndarray) -> np.ndarray:
    """Map scalar values in [0, 1] to RGB using a jet-like colormap.
        Blue = 0 (poor), red = 1 (good).
    """
    t = np.clip(t, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(t - 0.75) * 4, 0.0, 1.0)
    g = np.clip(1.5 - np.abs(t - 0.50) * 4, 0.0, 1.0)
    b = np.clip(1.5 - np.abs(t - 0.25) * 4, 0.0, 1.0)
    return np.stack([r, g, b], axis=1).astype(np.float32)
