"""Wilder's ADX with +DI and -DI, three outputs.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class ADX_talib:

    def __init__(self, window_size=14):
        self.window_size = window_size

    def __call__(self, high, low, close):
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)
        n = self.window_size
        # screamer's column order is (+DI, -DI, ADX). Getting this wrong shows
        # up as a ~100% mismatch on all three columns rather than a subtle one.
        return np.column_stack([
            talib.PLUS_DI(h, l, c, timeperiod=n),
            talib.MINUS_DI(h, l, c, timeperiod=n),
            talib.ADX(h, l, c, timeperiod=n),
        ])
