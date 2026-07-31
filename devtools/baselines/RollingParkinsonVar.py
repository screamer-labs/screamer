"""Parkinson (1980) range-based variance, rolling mean of the per-bar term.

The per-bar kernel comes from QuantLib's ParkinsonSigma, so the formula and its
1/(4 ln 2) constant are verified against an independent implementation rather
than against a transcription of the same expression.
"""
import numpy as np
import QuantLib as ql

from ._ohlc_vol import quantlib_per_bar_sigma
from ._windowing import rolling_mean


class RollingParkinsonVar_quantlib:

    def __init__(self, window_size=20):
        self.window_size = window_size

    def _per_bar_variance(self, high, low):
        # ParkinsonSigma reads only high and low; open and close are ignored.
        sigma = quantlib_per_bar_sigma(ql.ParkinsonSigma(1.0), high, high, low, low)
        return sigma ** 2

    def __call__(self, high, low):
        return rolling_mean(self._per_bar_variance(np.asarray(high, dtype=float),
                                                   np.asarray(low, dtype=float)),
                            self.window_size)


class RollingParkinsonVol_quantlib(RollingParkinsonVar_quantlib):
    """sqrt of the variance form."""

    def __call__(self, high, low):
        return np.sqrt(super().__call__(high, low))
