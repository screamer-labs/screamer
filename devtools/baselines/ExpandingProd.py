import numpy as np


class ExpandingProd_numpy:
    def __init__(self):
        pass

    def __call__(self, array):
        return np.cumprod(array)
