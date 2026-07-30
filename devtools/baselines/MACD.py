"""MACD line, signal line, and histogram.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class MACD_talib:

    def __init__(self, fast=12, slow=26, signal=9):
        self.fast, self.slow, self.signal = fast, slow, signal

    def __call__(self, x):
        macd, sig, hist = talib.MACD(np.asarray(x, dtype=float), fastperiod=self.fast,
                                     slowperiod=self.slow, signalperiod=self.signal)
        return np.column_stack([macd, sig, hist])
