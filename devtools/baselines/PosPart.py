import numpy as np

class PosPart_numpy:
    def __call__(self, array):
        array = np.asarray(array, dtype=np.float64)
        return np.where(np.isnan(array), array, np.maximum(array, 0.0))
