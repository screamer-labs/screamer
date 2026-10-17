"""Time latch: a nonzero input is held for n steps total, then `cooldown`
steps in which nonzero inputs are ignored.

The trigger step counts as step 1, so a nonzero input at t is emitted at
t, t+1, ..., t+n-1. When no hold is active the output is `release`. A NaN
input gives NaN and changes nothing.
"""
import math

import numpy as np


class Hold_numpy:

    def __init__(self, n=3, release=0.0, cooldown=0):
        if n < 1 or cooldown < 0:
            raise ValueError("n must be >= 1 and cooldown >= 0")
        self.n, self.release, self.cooldown = n, release, cooldown

    def __call__(self, x):
        held, remaining, cool = self.release, 0, 0
        out = np.empty(len(x), dtype=float)
        for i, v in enumerate(np.asarray(x, dtype=float)):
            if math.isnan(v):
                out[i] = math.nan
            elif remaining == 0 and cool > 0:
                cool -= 1
                out[i] = self.release
            elif v != 0.0:
                held, remaining = v, self.n - 1
                if remaining == 0:
                    cool = self.cooldown
                out[i] = v
            elif remaining > 0:
                remaining -= 1
                if remaining == 0:
                    cool = self.cooldown
                out[i] = held
            else:
                out[i] = self.release
        return out
