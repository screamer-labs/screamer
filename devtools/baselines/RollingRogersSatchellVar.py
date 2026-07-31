"""Rogers-Satchell (1991) drift-robust range-based variance.

    per-bar: ln(H/C) ln(H/O) + ln(L/C) ln(L/O)

Not in QuantLib, so this is transcribed from the documented formula. The
anchor against ground truth is tests/test_ohlc_volatility.py, which requires it
to recover the sigma of a simulated path *with* drift, which is the property
this estimator exists for.
"""
import numpy as np

from ._windowing import rolling_mean


class RollingRogersSatchellVar_numpy:

    def __init__(self, window_size=20):
        self.window_size = window_size

    def _per_bar_variance(self, o, h, l, c):
        return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

    def __call__(self, open_, high, low, close):
        return rolling_mean(self._per_bar_variance(
            np.asarray(open_, dtype=float), np.asarray(high, dtype=float),
            np.asarray(low, dtype=float), np.asarray(close, dtype=float)),
            self.window_size)


class RollingRogersSatchellVol_numpy(RollingRogersSatchellVar_numpy):

    def __call__(self, open_, high, low, close):
        return np.sqrt(super().__call__(open_, high, low, close))
