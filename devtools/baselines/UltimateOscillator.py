"""Williams (1976) three-period weighted oscillator.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class UltimateOscillator_talib:

    def __init__(self, period1=7, period2=14, period3=28):
        self.p1, self.p2, self.p3 = period1, period2, period3

    def __call__(self, high, low, close):
        return talib.ULTOSC(np.asarray(high, dtype=float), np.asarray(low, dtype=float),
                            np.asarray(close, dtype=float),
                            timeperiod1=self.p1, timeperiod2=self.p2, timeperiod3=self.p3)
