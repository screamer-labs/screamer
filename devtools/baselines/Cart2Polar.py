"""Convert (x, y) to (r, theta), returned as an (N, 2) array."""
import numpy as np


class Cart2Polar_numpy:

    def __call__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.column_stack([np.hypot(x, y), np.arctan2(y, x)])
