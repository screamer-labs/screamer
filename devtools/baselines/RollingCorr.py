"""Rolling Pearson correlation of two parallel streams."""
import numpy as np
import pandas as pd


class RollingCorr_pandas:

    def __init__(self, window_size=20, start_policy="strict"):
        self.window_size = window_size

    def __call__(self, x, y):
        a = pd.Series(np.asarray(x, dtype=float))
        b = pd.Series(np.asarray(y, dtype=float))
        return a.rolling(self.window_size).corr(b).to_numpy()
