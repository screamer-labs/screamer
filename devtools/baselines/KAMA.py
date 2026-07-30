"""Kaufman adaptive moving average.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class KAMA_talib:

    def __init__(self, window_size=10, fast=2, slow=30):
        self.window_size, self.fast, self.slow = window_size, fast, slow

    def __call__(self, x):
        return talib.KAMA(np.asarray(x, dtype=float), timeperiod=self.window_size)
