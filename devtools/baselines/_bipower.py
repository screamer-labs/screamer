"""Bipower and realized variance of trailing windows of returns."""
import math

import numpy as np

from ._windowing import ignoring_nan, trailing


def bipower_and_realized(x, window, start_policy):
    """(BV, RV) per step, NaN where the policy gives no output."""
    def run(v):
        bv, rv = [], []
        for w in trailing(v, window, start_policy, minimum=2):
            if w is None:
                bv.append(np.nan)
                rv.append(np.nan)
                continue
            n = len(w)
            bv.append(math.pi / 2 * np.sum(np.abs(w[1:]) * np.abs(w[:-1])) * n / (n - 1))
            rv.append(np.sum(w * w))
        return np.column_stack([bv, rv])
    return ignoring_nan(x, run)
