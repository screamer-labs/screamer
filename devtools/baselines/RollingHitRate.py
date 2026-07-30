"""Fraction of strictly-positive samples in the trailing window."""
import numpy as np
import pandas as pd


class RollingHitRate_pandas:

    def __init__(self, window_size=252):
        self.window_size = window_size

    def __call__(self, x):
        s = pd.Series((np.asarray(x, dtype=float) > 0.0).astype(float))
        return s.rolling(self.window_size).mean().to_numpy()
