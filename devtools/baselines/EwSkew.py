"""Exponentially weighted adjusted skewness.

Written from the standard estimator rather than from screamer's implementation:

    G1 = sqrt(n(n-1))/(n-2) * m3 / m2^(3/2)

with `m2` and `m3` the population (biased) central moments under the EW weights,
and `n` the effective sample size `(sum w)^2 / sum w^2`. Under equal weights this
reduces to `scipy.stats.skew(..., bias=False)`, which is what anchors it; see
`tests/test_ew_moments.py`.

The previous version standardized each point by the *running* std inside the EW
average instead of dividing the central moment once, so it computed neither this
quantity nor any other named one.
"""
import numpy as np


def alpha_from(com=None, span=None, halflife=None, alpha=None):
    if alpha is not None:
        return alpha
    if com is not None:
        return 1.0 / (1.0 + com)
    if span is not None:
        return 2.0 / (span + 1.0)
    if halflife is not None:
        return 1.0 - np.exp(-np.log(2.0) / halflife)
    raise ValueError("One of com, span, halflife, or alpha must be provided.")


def ew_central_moments(x, alpha):
    """Running (n_eff, m2, m3, m4) under EW weights, one row per sample."""
    w = 1.0 - alpha
    sx = sxx = sxxx = sxxxx = sw = sw2 = 0.0
    out = []
    for v in np.asarray(x, dtype=float):
        sx *= w; sxx *= w; sxxx *= w; sxxxx *= w
        sw *= w; sw2 *= w * w
        sx += v; sxx += v * v; sxxx += v ** 3; sxxxx += v ** 4
        sw += 1.0; sw2 += 1.0
        mean = sx / sw
        m2 = sxx / sw - mean * mean
        m3 = sxxx / sw - 3 * mean * (sxx / sw) + 2 * mean ** 3
        m4 = sxxxx / sw - 4 * mean * (sxxx / sw) + 6 * mean * mean * (sxx / sw) - 3 * mean ** 4
        out.append((sw * sw / sw2, m2, m3, m4))
    return np.array(out)


class EwSkew_numpy:

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.alpha = alpha_from(com, span, halflife, alpha)

    def __call__(self, x):
        n, m2, m3, _ = ew_central_moments(x, self.alpha).T
        with np.errstate(invalid="ignore", divide="ignore"):
            g1 = m3 / m2 ** 1.5
            out = np.sqrt(n * (n - 1.0)) / (n - 2.0) * g1
        out[n <= 2.0] = np.nan
        return out
