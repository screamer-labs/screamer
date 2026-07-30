"""Running minimum from t=0."""
import numpy as np


class CumMin_numpy:

    def __call__(self, x):
        return np.minimum.accumulate(np.asarray(x, dtype=float))
