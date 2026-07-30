"""Time latch: a nonzero input is held for n bars total.

The trigger bar counts as bar 1, so a nonzero input at t is emitted at
t, t+1, ..., t+n-1. When no hold is active the output is `release`.
"""
import numpy as np


class Hold_numpy:

    def __init__(self, n=3, release=0.0):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n, self.release = n, release

    def __call__(self, x):
        held, remaining = self.release, 0
        out = np.empty(len(x), dtype=float)
        for i, v in enumerate(np.asarray(x, dtype=float)):
            if np.isfinite(v) and v != 0.0:
                held, remaining = v, self.n - 1
                out[i] = held
            elif remaining > 0:
                remaining -= 1
                out[i] = held
            else:
                out[i] = self.release
        return out
