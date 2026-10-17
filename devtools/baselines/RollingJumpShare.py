import numpy as np

from ._bipower import bipower_and_realized


class RollingJumpShare_numpy:
    def __init__(self, window_size=20, start_policy="strict"):
        self.w, self.sp = window_size, start_policy

    def __call__(self, x):
        bv, rv = bipower_and_realized(x, self.w, self.sp).T
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.maximum(0.0, 1.0 - bv / rv)
        share[rv == 0.0] = 0.0
        share[np.isnan(bv) | np.isnan(rv)] = np.nan
        return share
