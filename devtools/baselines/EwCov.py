"""EW covariance of two streams, from pandas ewm().cov()."""
import numpy as np
import pandas as pd


class EwCov_pandas:

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.kw = dict(com=com, span=span, halflife=halflife, alpha=alpha)

    def __call__(self, x, y):
        a = pd.Series(np.asarray(x, dtype=float))
        b = pd.Series(np.asarray(y, dtype=float))
        return a.ewm(**self.kw).cov(b).to_numpy()
