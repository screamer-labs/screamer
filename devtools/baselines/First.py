import numpy as np


class First_numpy:
    def __call__(self, array):
        array = np.asarray(array, dtype=float)
        result = np.empty_like(array)
        first_val = np.nan
        has_value = False
        for i, x in enumerate(array):
            if np.isnan(x):
                result[i] = np.nan
            else:
                if not has_value:
                    first_val = x
                    has_value = True
                result[i] = first_val
        return result
