"""Pass-through, y = x.

Written from the documented definition, not from screamer's implementation.
"""
import numpy as np


class Identity_numpy:

    def __call__(self, x):
        return np.asarray(x, dtype=float).copy()
