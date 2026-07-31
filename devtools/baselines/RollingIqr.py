"""Q3 minus Q1 over the trailing window, linear interpolation."""
import numpy as np
import pandas as pd


class RollingIqr_pandas:

    def __init__(self, window_size=20):
        self.window_size = window_size

    def __call__(self, x):
        s = pd.Series(np.asarray(x, dtype=float)).rolling(self.window_size)
        return (s.quantile(0.75) - s.quantile(0.25)).to_numpy()
