"""Per-bar true range, accounting for the overnight gap.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class TrueRange_talib:

    def __call__(self, high, low, close):
        return talib.TRANGE(np.asarray(high, dtype=float), np.asarray(low, dtype=float),
                            np.asarray(close, dtype=float))
