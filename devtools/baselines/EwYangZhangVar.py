"""Yang-Zhang (2000) variance, exponentially weighted.

    sigma^2_o  = EW variance of overnight log returns ln(O_t / C_{t-1})
    sigma^2_c  = EW variance of open-to-close log returns ln(C_t / O_t)
    sigma^2_RS = EW mean of the per-bar Rogers-Satchell term
    k          = 0.34 / (1.34 + (n_eff+1)/(n_eff-1))   n_eff of the overnight leg
    sigma^2_YZ = sigma^2_o + k sigma^2_c + (1-k) sigma^2_RS

Not in QuantLib, so transcribed from the documented formula. The variances and
mean come from pandas ewm(); k uses the effective sample size of the overnight
leg. The anchor is tests/test_ohlc_volatility.py, which requires the EW form to
recover the sigma of a simulated path with both drift and overnight gaps.
"""
import numpy as np
import pandas as pd


def _alpha(com=None, span=None, halflife=None, alpha=None):
    if alpha is not None:
        return alpha
    if com is not None:
        return 1.0 / (1.0 + com)
    if span is not None:
        return 2.0 / (span + 1.0)
    if halflife is not None:
        return 1.0 - np.exp(-np.log(2.0) / halflife)
    raise ValueError("one of com/span/halflife/alpha")


class EwYangZhangVar_numpy:

    def __init__(self, com=None, span=None, halflife=None, alpha=None):
        self.kw = dict(com=com, span=span, halflife=halflife, alpha=alpha)
        self.a = _alpha(**self.kw)

    def __call__(self, open_, high, low, close):
        o = np.asarray(open_, dtype=float)
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        c = np.asarray(close, dtype=float)
        N = len(c)

        # Overnight leg: a clean series r_1..r_{N-1}, aligned back to index 1..N-1.
        overnight = np.log(o[1:] / c[:-1])
        var_o = np.full(N, np.nan)
        var_o[1:] = pd.Series(overnight).ewm(**self.kw).var().to_numpy()

        # Effective sample size of the overnight leg (same recursion as the op).
        om, om2 = 1.0 - self.a, (1.0 - self.a) ** 2
        n_eff = np.full(N, np.nan)
        sw = sw2 = 0.0
        for j in range(N - 1):
            sw = om * sw + 1.0
            sw2 = om2 * sw2 + 1.0
            n_eff[j + 1] = sw * sw / sw2
        with np.errstate(divide="ignore", invalid="ignore"):
            k = 0.34 / (1.34 + (n_eff + 1.0) / (n_eff - 1.0))

        # Open-to-close variance and Rogers-Satchell mean, from index 0.
        open_close = np.log(c / o)
        rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
        var_c = pd.Series(open_close).ewm(**self.kw).var().to_numpy()
        mean_rs = pd.Series(rs).ewm(**self.kw).mean().to_numpy()

        return var_o + k * var_c + (1.0 - k) * mean_rs


class EwYangZhangVol_numpy(EwYangZhangVar_numpy):

    def __call__(self, open_, high, low, close):
        return np.sqrt(super().__call__(open_, high, low, close))
