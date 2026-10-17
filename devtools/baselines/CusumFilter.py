"""Symmetric CUSUM event filter (Lopez de Prado 2018, section 2.5.2.1),
written as a plain loop over the input."""
import math

import numpy as np


class CusumFilter_numpy:
    def __init__(self, threshold=1.0, drift=0.0):
        self.h, self.drift = threshold, drift

    def __call__(self, x):
        pos = neg = 0.0
        out = np.empty(len(x), dtype=float)
        for i, v in enumerate(np.asarray(x, dtype=float)):
            if math.isnan(v):
                out[i] = math.nan
                continue
            pos = max(0.0, pos + v - self.drift)
            neg = min(0.0, neg + v + self.drift)
            if neg < -self.h:
                out[i], neg = -1.0, 0.0
            elif pos > self.h:
                out[i], pos = 1.0, 0.0
            else:
                out[i] = 0.0
        return out
