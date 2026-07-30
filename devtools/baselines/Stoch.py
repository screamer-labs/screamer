"""Stochastic oscillator, %K and %D.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class Stoch_talib:

    def __init__(self, fastk_period=14, smooth_k=3, d=3):
        self.fastk_period, self.smooth_k, self.d = fastk_period, smooth_k, d

    def __call__(self, high, low, close):
        k, dd = talib.STOCH(
            np.asarray(high, dtype=float), np.asarray(low, dtype=float),
            np.asarray(close, dtype=float),
            fastk_period=self.fastk_period,
            slowk_period=self.smooth_k, slowk_matype=0,
            slowd_period=self.d, slowd_matype=0,
        )
        return np.column_stack([k, dd])
