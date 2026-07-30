"""Hysteresis comparator with a dead band.

Output goes to 1.0 above `upper` and to 0.0 below `lower`. Inside the dead band
the previous state is held. Before the first crossing the output is `initial`.
"""
import numpy as np


class SchmittTrigger_numpy:

    def __init__(self, lower=-1.0, upper=1.0, initial=0.0):
        if not lower < upper:
            raise ValueError("lower must be strictly less than upper")
        self.lower, self.upper, self.initial = lower, upper, initial

    def __call__(self, x):
        state = self.initial
        out = np.empty(len(x), dtype=float)
        for i, v in enumerate(np.asarray(x, dtype=float)):
            if np.isnan(v):
                out[i] = np.nan
                continue
            if v > self.upper:
                state = 1.0
            elif v < self.lower:
                state = 0.0
            out[i] = state
        return out
