"""Chaikin Accumulation/Distribution line.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class AD_talib:

    def __call__(self, high, low, close, volume):
        return talib.AD(np.asarray(high, dtype=float), np.asarray(low, dtype=float),
                        np.asarray(close, dtype=float), np.asarray(volume, dtype=float))
