"""x[t] minus its trailing rolling mean."""
import numpy as np
import pandas as pd


class Detrend_pandas:

    def __init__(self, window_size=20, start_policy="strict"):
        self.window_size = window_size
        self.start_policy = start_policy

    def __call__(self, x):
        s = pd.Series(np.asarray(x, dtype=float))
        return (s - s.rolling(self.window_size).mean()).to_numpy()
