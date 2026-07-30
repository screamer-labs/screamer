"""Relative Strength Index, Wilder smoothing.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class RollingRSI_talib:

    def __init__(self, window_size=14, method="wilder", start_policy="strict"):
        self.window_size, self.method = window_size, method

    def __call__(self, x):
        return talib.RSI(np.asarray(x, dtype=float), timeperiod=self.window_size)
