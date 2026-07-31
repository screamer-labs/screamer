"""Garman-Klass (1980) range-based variance, rolling mean of the per-bar term.

The per-bar kernel comes from QuantLib's GarmanKlassSigma5, which is the
paper's simplified estimator

    0.5 * ln(H/L)^2 - (2 ln2 - 1) * ln(C/O)^2

and the one screamer implements. QuantLib also exposes Sigma1, Sigma3, Sigma4
and Sigma6: Sigma1/3/6 add an overnight term driven by a market-open fraction,
and Sigma4 is a higher-order approximation of the same estimand that differs in
the fourth significant figure. Sigma5 is the match.
"""
import numpy as np
import QuantLib as ql

from ._ohlc_vol import quantlib_per_bar_sigma
from ._windowing import rolling_mean


class RollingGarmanKlassVar_quantlib:

    def __init__(self, window_size=20):
        self.window_size = window_size

    def _per_bar_variance(self, o, h, l, c):
        return quantlib_per_bar_sigma(ql.GarmanKlassSigma5(1.0), o, h, l, c) ** 2

    def __call__(self, open_, high, low, close):
        return rolling_mean(self._per_bar_variance(
            np.asarray(open_, dtype=float), np.asarray(high, dtype=float),
            np.asarray(low, dtype=float), np.asarray(close, dtype=float)),
            self.window_size)


class RollingGarmanKlassVol_quantlib(RollingGarmanKlassVar_quantlib):

    def __call__(self, open_, high, low, close):
        return np.sqrt(super().__call__(open_, high, low, close))
