"""Hull's moving average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
import numpy as np
import pandas as pd


def _wma(series, n):
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    return series.rolling(n).apply(lambda v: float(v @ w), raw=True)


class HullMA_pandas:

    def __init__(self, window_size=20):
        self.n = window_size

    def __call__(self, x):
        s = pd.Series(np.asarray(x, dtype=float))
        half = int(self.n // 2)
        root = int(np.sqrt(self.n))
        raw = 2.0 * _wma(s, half) - _wma(s, self.n)
        return _wma(raw, root).to_numpy()
