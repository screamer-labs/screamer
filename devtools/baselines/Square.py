"""x squared.

Written from the documented definition, not from screamer's implementation.
"""
import numpy as np


class Square_numpy:

    def __call__(self, x):
        return np.asarray(x, dtype=float) ** 2
