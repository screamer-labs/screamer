"""Window-offset of the trailing-window minimum.

TA-Lib is the reference this operator's page cites as its definition.
"""
import numpy as np
import talib


class RollingArgmin_talib:
    """TA-Lib returns the absolute array index; screamer returns the offset
    within the window, 0 being the oldest sample. The conversion subtracts the
    index of the window's first element."""

    def __init__(self, window_size=20):
        self.window_size = window_size

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        absolute = talib.MININDEX(x, timeperiod=self.window_size)
        window_start = np.arange(len(x)) - self.window_size + 1
        return absolute - window_start
