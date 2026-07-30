"""1.0 if a == b, else 0.0.

Written from the documented definition.
"""
import numpy as np


class Equal_numpy:

    def __call__(self, x, y):
        return (np.asarray(x, dtype=float) == np.asarray(y, dtype=float)).astype(float)
