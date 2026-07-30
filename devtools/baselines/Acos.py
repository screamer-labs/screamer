"""Inverse cosine, radians.

Written from the documented definition, not from screamer's implementation.
"""
import numpy as np


class Acos_numpy:

    def __call__(self, x):
        return np.arccos(x)
