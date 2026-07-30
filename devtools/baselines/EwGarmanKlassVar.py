"""Garman-Klass (1980) variance, exponentially weighted.

Per-bar kernel from QuantLib's GarmanKlassSigma5, the paper's simplified form
and the one screamer implements. See RollingGarmanKlassVar for why Sigma4 and
the gap-aware Sigma1/3/6 are not the right comparison.
"""
import numpy as np
import pandas as pd

from ._ohlc_vol import quantlib_per_bar_sigma


class EwGarmanKlassVar_quantlib:

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.kw = dict(com=com, span=span, halflife=halflife, alpha=alpha)

    def _per_bar_variance(self, o, h, l, c):
        import QuantLib as ql
        return quantlib_per_bar_sigma(
            ql.GarmanKlassSigma5(1.0),
            np.asarray(o, dtype=float), np.asarray(h, dtype=float),
            np.asarray(l, dtype=float), np.asarray(c, dtype=float),
        ) ** 2

    def __call__(self, open_, high, low, close):
        var = self._per_bar_variance(open_, high, low, close)
        return pd.Series(var).ewm(**self.kw).mean().to_numpy()


class EwGarmanKlassVol_quantlib(EwGarmanKlassVar_quantlib):

    def __call__(self, open_, high, low, close):
        return np.sqrt(super().__call__(open_, high, low, close))
