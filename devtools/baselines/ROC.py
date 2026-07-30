"""100 * (x[t] / x[t-k] - 1), TA-Lib's ROC."""
import numpy as np


class ROC_numpy:

    def __init__(self, window_size=10):
        self.k = window_size

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        out = np.full(len(x), np.nan)
        out[self.k:] = 100.0 * (x[self.k:] / x[:-self.k] - 1.0)
        return out
