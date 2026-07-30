"""Trailing-window max minus min."""
import numpy as np
import pandas as pd


class RollingRange_pandas:

    def __init__(self, window_size=20):
        self.window_size = window_size

    def __call__(self, x):
        s = pd.Series(np.asarray(x, dtype=float))
        r = s.rolling(self.window_size)
        return (r.max() - r.min()).to_numpy()
