"""Inverse tangent, radians.

Written from the documented definition, not from screamer's implementation.
"""
import numpy as np


class Atan_numpy:

    def __call__(self, x):
        return np.arctan(x)
