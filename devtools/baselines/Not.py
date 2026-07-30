"""0.0 where the input is nonzero, 1.0 where it is zero.

Written from the documented definition, not from screamer's implementation.
"""
import numpy as np


class Not_numpy:

    def __call__(self, x):
        return (np.asarray(x, dtype=float) == 0.0).astype(float)
