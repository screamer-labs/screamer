"""Wilder-smoothed average true range.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class ATR_talib:

    def __init__(self, window_size=14):
        self.window_size = window_size

    def __call__(self, high, low, close):
        return talib.ATR(np.asarray(high, dtype=float), np.asarray(low, dtype=float),
                          np.asarray(close, dtype=float), timeperiod=self.window_size)
