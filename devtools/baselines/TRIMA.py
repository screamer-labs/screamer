"""Triangular moving average: an SMA of an SMA, using TA-Lib's window split.

    odd  n: n_inner = n_outer = (n + 1) / 2
    even n: n_inner = n/2 + 1, n_outer = n/2

In both cases n_inner + n_outer - 1 == n.
"""
import numpy as np
import pandas as pd


class TRIMA_pandas:

    def __init__(self, window_size=20):
        n = window_size
        if n % 2:
            self.inner = self.outer = (n + 1) // 2
        else:
            self.inner, self.outer = n // 2 + 1, n // 2

    def __call__(self, x):
        s = pd.Series(np.asarray(x, dtype=float))
        return s.rolling(self.inner).mean().rolling(self.outer).mean().to_numpy()
