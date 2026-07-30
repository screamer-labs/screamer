"""Signed angle of (x, y) from the positive x-axis.

Written from the documented definition.
"""
import numpy as np


class Atan2_numpy:

    def __call__(self, x, y):
        return np.arctan2(x, y)
