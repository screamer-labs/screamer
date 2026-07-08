import numpy as np


class ExpandingSlope_numpy:
    def __init__(self):
        pass

    def __call__(self, array):
        array = np.asarray(array, dtype=float)
        out = np.full(array.shape, np.nan)
        for t in range(1, len(array)):
            x = np.arange(t + 1)
            out[t] = np.polyfit(x, array[: t + 1], 1)[0]
        return out
