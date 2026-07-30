"""Convert (r, theta) to (x, y), returned as an (N, 2) array."""
import numpy as np


class Polar2Cart_numpy:

    def __call__(self, r, theta):
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
