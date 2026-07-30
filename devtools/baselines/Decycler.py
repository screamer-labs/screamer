import numpy as np
from scipy.signal import lfilter


class Decycler_ehlers:
    def __init__(self, period=None, cutoff=None):
        if (period is None) == (cutoff is None):
            raise ValueError("Provide exactly one of period or cutoff.")
        p = period if period is not None else 2.0 / cutoff
        w = 2 * np.pi / p
        alpha = (np.cos(w) + np.sin(w) - 1) / np.cos(w)
        self.b = [alpha / 2, alpha / 2]
        self.a = [1.0, -(1 - alpha)]

    def __call__(self, array):
        return lfilter(self.b, self.a, np.asarray(array, float))
