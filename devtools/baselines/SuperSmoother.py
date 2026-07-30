import numpy as np
from scipy.signal import lfilter


class SuperSmoother_ehlers:
    def __init__(self, period=None, cutoff=None):
        if (period is None) == (cutoff is None):
            raise ValueError("Provide exactly one of period or cutoff.")
        p = period if period is not None else 2.0 / cutoff
        a1 = np.exp(-np.sqrt(2) * np.pi / p)
        b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / p)
        c2, c3 = b1, -a1 * a1
        c1 = 1 - c2 - c3
        self.b = [c1 / 2, c1 / 2]
        self.a = [1.0, -c2, -c3]

    def __call__(self, array):
        return lfilter(self.b, self.a, np.asarray(array, float))
