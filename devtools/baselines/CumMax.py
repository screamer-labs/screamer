"""Running maximum from t=0."""
import numpy as np


class CumMax_numpy:

    def __call__(self, x):
        return np.maximum.accumulate(np.asarray(x, dtype=float))
