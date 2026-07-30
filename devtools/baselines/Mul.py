"""Elementwise product, x * y.

Written from the documented definition.
"""
import numpy as np


class Mul_numpy:

    def __call__(self, x, y):
        return np.asarray(x, dtype=float) * np.asarray(y, dtype=float)
