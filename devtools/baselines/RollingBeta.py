"""cov(x, y) / var(y): the regression slope of x on y."""
import numpy as np
import pandas as pd


class RollingBeta_pandas:

    def __init__(self, window_size=20, start_policy="strict"):
        self.window_size = window_size

    def __call__(self, x, y):
        a = pd.Series(np.asarray(x, dtype=float))
        b = pd.Series(np.asarray(y, dtype=float))
        return (a.rolling(self.window_size).cov(b)
                / b.rolling(self.window_size).var()).to_numpy()
