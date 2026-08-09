"""EW regression intercept: ewmean(target) - EwBeta(target, regressor) * ewmean(regressor).

Built from pandas ewm() so the weighting convention comes from pandas.
"""
import numpy as np
import pandas as pd


class EwAlpha_pandas:

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.kw = dict(com=com, span=span, halflife=halflife, alpha=alpha)

    def __call__(self, target, regressor):
        t = pd.Series(np.asarray(target, dtype=float))
        r = pd.Series(np.asarray(regressor, dtype=float))
        beta = t.ewm(**self.kw).cov(r) / r.ewm(**self.kw).var()
        return (t.ewm(**self.kw).mean() - beta * r.ewm(**self.kw).mean()).to_numpy()
