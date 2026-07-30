"""FIR filter with user-supplied taps.

    y[t] = sum_k taps[k] * x[t - k]

taps[0] multiplies the current sample. The first L-1 outputs are NaN, since the
filter is not fully defined until L samples have arrived.
"""
import numpy as np


class MovingAverage_numpy:

    def __init__(self, taps=(0.25, 0.5, 0.25)):
        self.taps = np.asarray(taps, dtype=float)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        out = np.convolve(x, self.taps, mode="full")[:len(x)]
        out[:len(self.taps) - 1] = np.nan
        return out
