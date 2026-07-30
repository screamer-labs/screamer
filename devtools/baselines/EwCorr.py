"""EW Pearson correlation of two streams, from pandas ewm().corr()."""
import numpy as np
import pandas as pd


class EwCorr_pandas:

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.kw = dict(com=com, span=span, halflife=halflife, alpha=alpha)

    def __call__(self, x, y):
        a = pd.Series(np.asarray(x, dtype=float))
        b = pd.Series(np.asarray(y, dtype=float))
        return a.ewm(**self.kw).corr(b).to_numpy()
