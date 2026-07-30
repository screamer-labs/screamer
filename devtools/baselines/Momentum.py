"""x[t] - x[t-k], TA-Lib's MOM."""
import numpy as np


class Momentum_numpy:

    def __init__(self, window_size=10, start_policy="strict"):
        self.k = window_size
        self.start_policy = start_policy

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        out = np.full(len(x), np.nan)
        out[self.k:] = x[self.k:] - x[:-self.k]
        return out
