"""(close - open) / (high - low) per bar.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class BOP_talib:

    def __call__(self, open_, high, low, close):
        return talib.BOP(np.asarray(open_, dtype=float), np.asarray(high, dtype=float),
                         np.asarray(low, dtype=float), np.asarray(close, dtype=float))
