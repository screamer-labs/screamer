"""Running product from t=0."""
import numpy as np


class CumProd_numpy:

    def __call__(self, x):
        return np.cumprod(np.asarray(x, dtype=float))
