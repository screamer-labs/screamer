"""Bandt-Pompe permutation entropy of the ordinal patterns ending in a
trailing window, normalized by log(order!). A pattern is the stable argsort
of order values spaced delay apart, so ties rank by order of appearance."""
import math
from collections import Counter

import numpy as np

from ._windowing import ignoring_nan


class RollingPermutationEntropy_numpy:
    def __init__(self, window_size=100, order=3, delay=1, start_policy="strict"):
        self.w, self.m, self.tau, self.sp = window_size, order, delay, start_policy

    def _run(self, x):
        span = (self.m - 1) * self.tau
        pad = self.w + span if self.sp == "zero" else 0
        x = np.concatenate([np.zeros(pad), x])
        codes = [None] * len(x)
        for t in range(span, len(x)):
            codes[t] = tuple(np.argsort(x[t - span: t + 1: self.tau], kind="stable"))
        norm = math.log(math.factorial(self.m))
        out = np.full(len(x), np.nan)
        for t in range(span, len(x)):
            win = codes[max(span, t - self.w + 1): t + 1]
            if self.sp == "strict" and len(win) < self.w:
                continue
            p = np.array(list(Counter(win).values()), dtype=float) / len(win)
            out[t] = max(0.0, -np.sum(p * np.log(p)) / norm)
        return out[pad:]

    def __call__(self, x):
        return ignoring_nan(x, self._run)
