"""Lo-MacKinlay variance ratio of overlapping lag-sums over a trailing window
of returns, with the bias-corrected variances, and the heteroskedasticity-
robust z*(q) (Lo and MacKinlay 1988, equations 12, 13, 18 to 20)."""
import math

import numpy as np

from ._windowing import ignoring_nan, trailing


def variance_ratio(r, q, output):
    n = len(r)
    mu = r.mean()
    e = r - mu
    s2 = np.sum(e * e)
    if not s2 > 0.0:
        return math.nan
    var_a = s2 / (n - 1)
    sums = np.convolve(r, np.ones(q), mode="valid")          # the n - q + 1 overlapping q-sums
    m = q * (n - q + 1) * (1.0 - q / n)
    ratio = np.sum((sums - q * mu) ** 2) / m / var_a
    if output == "ratio":
        return ratio
    theta = sum((2.0 * (q - j) / q) ** 2 * n * np.sum(e[j:] ** 2 * e[:-j] ** 2) / s2 ** 2
                for j in range(1, q))
    return math.sqrt(n) * (ratio - 1.0) / math.sqrt(theta)


class RollingVarianceRatio_numpy:
    def __init__(self, window_size=256, lag=2, output="ratio", start_policy="strict"):
        self.w, self.q, self.output, self.sp = window_size, lag, output, start_policy

    def _run(self, x):
        return np.array([math.nan if w is None else variance_ratio(w, self.q, self.output)
                         for w in trailing(x, self.w, self.sp, minimum=2 * self.q + 1)])

    def __call__(self, x):
        return ignoring_nan(x, self._run)
