"""Euclidean distance sqrt(x^2 + y^2).

Written from the documented definition.
"""
import numpy as np


class Hypot_numpy:

    def __call__(self, x, y):
        return np.hypot(x, y)
