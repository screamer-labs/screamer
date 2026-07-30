"""1.0 if either input is nonzero, else 0.0.

Written from the documented definition.
"""
import numpy as np


class Or_numpy:

    def __call__(self, x, y):
        return ((np.asarray(x, dtype=float) != 0.0) | (np.asarray(y, dtype=float) != 0.0)).astype(float)
