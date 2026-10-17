import numpy as np

from ._seasonal import slot_windows


class RollingSeasonalStd_numpy:
    def __init__(self, period=24, window_size=20, start_policy="strict"):
        self.p, self.w, self.sp = period, window_size, start_policy

    def __call__(self, x):
        return np.array([np.nan if w is None else w.std(ddof=1)
                         for w in slot_windows(x, self.p, self.w, self.sp, 2)])
