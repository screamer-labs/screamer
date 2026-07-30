"""Money Flow Index: volume-weighted RSI on the typical price.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class MFI_talib:

    def __init__(self, window_size=14):
        self.window_size = window_size

    def __call__(self, high, low, close, volume):
        return talib.MFI(np.asarray(high, dtype=float), np.asarray(low, dtype=float),
                         np.asarray(close, dtype=float), np.asarray(volume, dtype=float),
                         timeperiod=self.window_size)
