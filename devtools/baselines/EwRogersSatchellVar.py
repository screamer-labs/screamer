"""Rogers-Satchell (1991) variance, exponentially weighted.

    per-bar: ln(H/C) ln(H/O) + ln(L/C) ln(L/O)

Not in QuantLib, so transcribed from the documented formula. The check against
ground truth is tests/test_ohlc_volatility.py, which requires the EW form to
recover a known sigma under drift, the property this estimator exists for.
"""
import numpy as np
import pandas as pd


class EwRogersSatchellVar_numpy:

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.kw = dict(com=com, span=span, halflife=halflife, alpha=alpha)

    def _per_bar_variance(self, o, h, l, c):
        o = np.asarray(o, dtype=float); h = np.asarray(h, dtype=float)
        l = np.asarray(l, dtype=float); c = np.asarray(c, dtype=float)
        return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

    def __call__(self, open_, high, low, close):
        var = self._per_bar_variance(open_, high, low, close)
        return pd.Series(var).ewm(**self.kw).mean().to_numpy()


class EwRogersSatchellVol_numpy(EwRogersSatchellVar_numpy):

    def __call__(self, open_, high, low, close):
        return np.sqrt(super().__call__(open_, high, low, close))
