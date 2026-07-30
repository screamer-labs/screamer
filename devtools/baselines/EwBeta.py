"""EW CAPM beta: cov(target, regressor) / var(regressor).

Built from pandas ewm().cov() and .var(), so the weighting convention comes
from pandas rather than from a transcription.
"""
import numpy as np
import pandas as pd


class EwBeta_pandas:

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.kw = dict(com=com, span=span, halflife=halflife, alpha=alpha)

    def __call__(self, target, regressor):
        y = pd.Series(np.asarray(target, dtype=float))
        x = pd.Series(np.asarray(regressor, dtype=float))
        return (y.ewm(**self.kw).cov(x) / x.ewm(**self.kw).var()).to_numpy()
