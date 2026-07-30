"""1.0 where the input is NaN, else 0.0.

Written from the documented definition, not from screamer's implementation.
"""
import numpy as np


class IsNan_numpy:

    def __call__(self, x):
        return np.isnan(x).astype(float)
