"""Trailing-window mean absolute deviation from the window mean.

    mad[t] = mean(|x_i - mean(window)|) over the window
"""
import numpy as np
import pandas as pd


class RollingMad_pandas:

    def __init__(self, window_size=20, start_policy="strict"):
        self.window_size = window_size
        self.start_policy = start_policy

    def __call__(self, x):
        s = pd.Series(np.asarray(x, dtype=float))
        return s.rolling(self.window_size).apply(
            lambda v: np.abs(v - v.mean()).mean(), raw=True
        ).to_numpy()
