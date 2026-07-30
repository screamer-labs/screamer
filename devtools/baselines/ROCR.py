"""x[t] / x[t-k], TA-Lib's ROCR."""
import numpy as np


class ROCR_numpy:

    def __init__(self, window_size=10):
        self.k = window_size

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        out = np.full(len(x), np.nan)
        out[self.k:] = x[self.k:] / x[:-self.k]
        return out
