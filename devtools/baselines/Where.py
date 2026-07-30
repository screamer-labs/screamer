"""Returns a where the mask is nonzero, b otherwise."""
import numpy as np


class Where_numpy:

    def __call__(self, mask, a, b):
        mask = np.asarray(mask, dtype=float)
        return np.where(mask != 0.0, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
