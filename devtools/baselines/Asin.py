"""Inverse sine, radians.

Written from the documented definition, not from screamer's implementation.
"""
import numpy as np


class Asin_numpy:

    def __call__(self, x):
        return np.arcsin(x)
