"""Rolling volume-weighted average of the typical price.

    tp = (high + low + close) / 3
    vwap = sum(tp * volume) / sum(volume)   over the trailing window
"""
import numpy as np
import pandas as pd


class RollingVWAP_pandas:

    def __init__(self, window_size=20):
        self.window_size = window_size

    def __call__(self, high, low, close, volume):
        tp = (np.asarray(high, dtype=float) + np.asarray(low, dtype=float)
              + np.asarray(close, dtype=float)) / 3.0
        v = np.asarray(volume, dtype=float)
        num = pd.Series(tp * v).rolling(self.window_size).sum()
        den = pd.Series(v).rolling(self.window_size).sum()
        return (num / den).to_numpy()
