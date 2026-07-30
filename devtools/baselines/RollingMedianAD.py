"""Rolling median absolute deviation, median(|x - median(window)|)."""
import numpy as np
import pandas as pd


class RollingMedianAD_pandas:

    def __init__(self, window_size=20, start_policy="strict"):
        self.window_size = window_size
        self.start_policy = start_policy

    def __call__(self, x):
        s = pd.Series(np.asarray(x, dtype=float))
        return s.rolling(self.window_size).apply(
            lambda v: np.median(np.abs(v - np.median(v))), raw=True
        ).to_numpy()
