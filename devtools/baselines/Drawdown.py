"""Running drawdown from the cumulative-since-inception peak.

    drawdown[t] = x[t] / cummax(x)[t] - 1

A flat or new-high series gives 0; a 30% loss from the prior peak gives -0.30.
"""
import numpy as np


class Drawdown_numpy:

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return x / np.maximum.accumulate(x) - 1.0
