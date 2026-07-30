"""x cubed.

Written from the documented definition, not from screamer's implementation.
"""
import numpy as np


class Cube_numpy:

    def __call__(self, x):
        return np.asarray(x, dtype=float) ** 3
