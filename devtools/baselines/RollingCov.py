"""Rolling sample covariance of two parallel streams (ddof=1)."""
import numpy as np
import pandas as pd


class RollingCov_pandas:

    def __init__(self, window_size=20, start_policy="strict"):
        self.window_size = window_size

    def __call__(self, x, y):
        a = pd.Series(np.asarray(x, dtype=float))
        b = pd.Series(np.asarray(y, dtype=float))
        return a.rolling(self.window_size).cov(b).to_numpy()
