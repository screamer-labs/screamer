"""Upside semideviation over a trailing window, sqrt(mean(max(r - mar, 0)^2))."""
import numpy as np

from ._windowing import ignoring_nan, trailing


class RollingUpsideDeviation_numpy:
    def __init__(self, window_size=20, mar=0.0, start_policy="strict"):
        self.w, self.mar, self.sp = window_size, mar, start_policy

    def _run(self, x):
        out = []
        for win in trailing(x, self.w, self.sp):
            out.append(np.nan if win is None
                       else np.sqrt(np.mean(np.maximum(win - self.mar, 0.0) ** 2)))
        return np.array(out)

    def __call__(self, x):
        return ignoring_nan(x, self._run)
