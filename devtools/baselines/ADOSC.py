"""Chaikin oscillator: fast EMA minus slow EMA of the A/D line.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class ADOSC_talib:

    def __init__(self, fast=3, slow=10):
        self.fast, self.slow = fast, slow

    def __call__(self, high, low, close, volume):
        return talib.ADOSC(np.asarray(high, dtype=float), np.asarray(low, dtype=float),
                           np.asarray(close, dtype=float), np.asarray(volume, dtype=float),
                           fastperiod=self.fast, slowperiod=self.slow)
