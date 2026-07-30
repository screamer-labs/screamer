"""Worst drawdown observed since the start: the running minimum of Drawdown."""
import numpy as np


class MaxDrawdown_numpy:

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return np.minimum.accumulate(x / np.maximum.accumulate(x) - 1.0)
