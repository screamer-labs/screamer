"""Linearly weighted moving average.

The newest sample carries weight w, the oldest weight 1, over the divisor
w(w+1)/2.
"""
import numpy as np
import pandas as pd


class WMA_pandas:

    def __init__(self, window_size=20, start_policy="strict"):
        self.window_size = window_size
        self.start_policy = start_policy

    def __call__(self, x):
        w = np.arange(1, self.window_size + 1, dtype=float)
        w /= w.sum()
        s = pd.Series(np.asarray(x, dtype=float))
        return s.rolling(self.window_size).apply(lambda v: float(v @ w), raw=True).to_numpy()
