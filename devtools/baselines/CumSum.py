"""Running sum from t=0."""
import numpy as np


class CumSum_numpy:

    def __call__(self, x):
        return np.cumsum(np.asarray(x, dtype=float))
