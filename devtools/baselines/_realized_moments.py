"""Realized skewness and kurtosis of trailing windows of returns, from the
formulas of Amaya, Christoffersen, Jacobs and Vasquez (2015)."""
import numpy as np

from ._windowing import ignoring_nan, trailing


def realized_moment(x, window, start_policy, power):
    """sqrt(N) sum r^3 / RV^1.5 (power 3) or N sum r^4 / RV^2 (power 4)."""
    def run(v):
        out = []
        for w in trailing(v, window, start_policy):
            rv = None if w is None else np.sum(w ** 2)
            if w is None or rv == 0.0:
                out.append(np.nan)
                continue
            n = len(w)
            out.append(n ** (power / 2 - 1) * np.sum(w ** power) / rv ** (power / 2))
        return out
    return ignoring_nan(x, run)
