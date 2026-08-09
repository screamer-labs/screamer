"""EW std of the spread target - EwBeta(target, regressor) * regressor.

Built from pandas ewm() so the weighting convention comes from pandas.
"""
import numpy as np
import pandas as pd


class EwResidualStd_pandas:

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.kw = dict(com=com, span=span, halflife=halflife, alpha=alpha)

    def __call__(self, target, regressor):
        t = pd.Series(np.asarray(target, dtype=float))
        r = pd.Series(np.asarray(regressor, dtype=float))
        beta = t.ewm(**self.kw).cov(r) / r.ewm(**self.kw).var()
        spread = t - beta * r
        return spread.ewm(**self.kw).std().to_numpy()
