"""Rate of change of a triple-smoothed EMA.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class TRIX_talib:

    def __init__(self, span=14):
        self.span = span

    def __call__(self, x):
        return talib.TRIX(np.asarray(x, dtype=float), timeperiod=self.span)
