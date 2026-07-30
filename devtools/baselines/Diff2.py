"""Second-order finite difference, y[t] = x[t] - 2x[t-1] + x[t-2].

Not the same as Diff(2), which is the lag-2 first difference.
"""
import numpy as np


class Diff2_numpy:

    def __init__(self, start_policy="strict"):
        self.start_policy = start_policy

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        out = np.full(len(x), np.nan)
        out[2:] = x[2:] - 2.0 * x[1:-1] + x[:-2]
        return out
