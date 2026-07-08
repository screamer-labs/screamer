import numpy as np


class Last_numpy:
    def __call__(self, array):
        array = np.asarray(array, dtype=float)
        result = np.empty_like(array)
        last_val = np.nan
        for i, x in enumerate(array):
            if np.isnan(x):
                result[i] = np.nan
            else:
                last_val = x
                result[i] = last_val
        return result
