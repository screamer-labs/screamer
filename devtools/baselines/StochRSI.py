"""Stochastic oscillator applied to RSI.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class StochRSI_talib:

    def __init__(self, rsi_period=14, stoch_period=14, smooth_k=1, d=3):
        self.rsi_period, self.stoch_period = rsi_period, stoch_period
        self.smooth_k, self.d = smooth_k, d

    def __call__(self, x):
        k, dd = talib.STOCHRSI(np.asarray(x, dtype=float), timeperiod=self.rsi_period,
                               fastk_period=self.stoch_period, fastd_period=self.d,
                               fastd_matype=0)
        return np.column_stack([k, dd])
