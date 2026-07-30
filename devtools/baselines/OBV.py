"""On-Balance Volume: signed cumulative volume.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class OBV_talib:

    def __call__(self, close, volume):
        return talib.OBV(np.asarray(close, dtype=float), np.asarray(volume, dtype=float))
