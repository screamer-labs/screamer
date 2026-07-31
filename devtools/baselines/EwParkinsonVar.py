"""Parkinson (1980) variance, exponentially weighted.

Same per-bar kernel as RollingParkinsonVar, taken from QuantLib's
ParkinsonSigma so the formula is checked against an independent
implementation; only the averaging differs, an EW mean in place of a rolling
one. `pandas.ewm(...).mean()` with its default `adjust=True` is the convention
the EwMean baseline already uses.
"""
import numpy as np
import QuantLib as ql
import pandas as pd

from ._ohlc_vol import quantlib_per_bar_sigma


class EwParkinsonVar_quantlib:

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.kw = dict(com=com, span=span, halflife=halflife, alpha=alpha)

    def _per_bar_variance(self, high, low):
        # ParkinsonSigma reads only the high and low fields.
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        return quantlib_per_bar_sigma(ql.ParkinsonSigma(1.0), h, h, l, l) ** 2

    def __call__(self, high, low):
        var = self._per_bar_variance(high, low)
        return pd.Series(var).ewm(**self.kw).mean().to_numpy()


class EwParkinsonVol_quantlib(EwParkinsonVar_quantlib):

    def __call__(self, high, low):
        return np.sqrt(super().__call__(high, low))
