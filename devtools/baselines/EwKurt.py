"""Exponentially weighted adjusted excess kurtosis.

Written from the standard estimator rather than from screamer's implementation:

    G2 = (n-1)/((n-2)(n-3)) * ((n+1) * m4/m2^2 - 3(n-1))

with `m2` and `m4` the population (biased) central moments under the EW weights,
and `n` the effective sample size. Under equal weights this reduces to
`scipy.stats.kurtosis(..., bias=False)`, which is what anchors it; see
`tests/test_ew_moments.py`.

The previous version standardized each point by the running std inside the EW
average, and then subtracted 3 from a correction that already returns *excess*
kurtosis. It returned -6 on normal data at every span.
"""
import numpy as np

from .EwSkew import alpha_from, ew_central_moments


class EwKurt_numpy:

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.alpha = alpha_from(com, span, halflife, alpha)

    def __call__(self, x):
        n, m2, _, m4 = ew_central_moments(x, self.alpha).T
        with np.errstate(invalid="ignore", divide="ignore"):
            g2 = m4 / (m2 * m2)
            out = (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * g2 - 3.0 * (n - 1.0))
        out[n <= 3.0] = np.nan
        return out
