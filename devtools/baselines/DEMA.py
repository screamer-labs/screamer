"""Mulloy's double EMA: 2*EMA - EMA(EMA)."""
import numpy as np
import pandas as pd


class DEMA_pandas:

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.kw = dict(com=com, span=span, halflife=halflife, alpha=alpha)

    def __call__(self, x):
        s = pd.Series(np.asarray(x, dtype=float))
        e1 = s.ewm(**self.kw).mean()
        e2 = e1.ewm(**self.kw).mean()
        return (2.0 * e1 - e2).to_numpy()
