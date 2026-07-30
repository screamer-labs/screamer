"""1.0 where the input is finite, else 0.0.

Written from the documented definition, not from screamer's implementation.
"""
import numpy as np


class IsFinite_numpy:

    def __call__(self, x):
        return np.isfinite(x).astype(float)
