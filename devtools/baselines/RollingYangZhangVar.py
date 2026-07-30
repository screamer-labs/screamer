"""Yang-Zhang (2000) drift- and gap-robust variance.

    sigma^2_o  = sample variance of overnight log returns ln(O_t / C_{t-1})
    sigma^2_c  = sample variance of open-to-close log returns ln(C_t / O_t)
    sigma^2_RS = mean of the per-bar Rogers-Satchell term
    k          = 0.34 / (1.34 + (n+1)/(n-1))
    sigma^2_YZ = sigma^2_o + k sigma^2_c + (1-k) sigma^2_RS

Not in QuantLib, so this is transcribed from the documented formula. The anchor
is tests/test_ohlc_volatility.py, which requires it to recover the sigma of a
simulated path with both drift and overnight gaps.
"""
import numpy as np
import pandas as pd


class RollingYangZhangVar_numpy:

    def __init__(self, window_size=20):
        self.window_size = window_size

    def __call__(self, open_, high, low, close):
        o = np.asarray(open_, dtype=float)
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)
        n = self.window_size

        overnight = np.full(len(c), np.nan)
        overnight[1:] = np.log(o[1:] / c[:-1])
        open_close = np.log(c / o)
        rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

        var_o = pd.Series(overnight).rolling(n).var(ddof=1)
        var_c = pd.Series(open_close).rolling(n).var(ddof=1)
        mean_rs = pd.Series(rs).rolling(n).mean()

        k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
        return (var_o + k * var_c + (1.0 - k) * mean_rs).to_numpy()


class RollingYangZhangVol_numpy(RollingYangZhangVar_numpy):

    def __call__(self, open_, high, low, close):
        return np.sqrt(super().__call__(open_, high, low, close))
