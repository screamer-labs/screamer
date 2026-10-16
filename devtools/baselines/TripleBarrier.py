"""Triple-barrier labels written as a plain loop over entries."""
import numpy as np
import pandas as pd


def _levels(p, upper, lower, unit, s):
    if unit == "fraction":
        U, L = p * (1 + upper), p * (1 - lower)
    elif unit == "price":
        U, L = p + upper, p - lower
    else:
        U, L = p * (1 + upper * s), p * (1 - lower * s)
    return (U if upper > 0 else np.inf), (L if lower > 0 else -np.inf)


def _sigma(price, vol_span, horizon):
    # One-step log returns; a return touching a NaN, infinite, or non-positive price is
    # missing and skipped by the weighting (ignore_na=True), as EwStd skips it.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(price[1:] / price[:-1])
    r[~((price[1:] > 0) & (price[:-1] > 0))] = np.nan
    r[~np.isfinite(r)] = np.nan
    s = pd.Series(np.r_[np.nan, r]).ewm(span=vol_span, ignore_na=True).std().to_numpy()
    return s * np.sqrt(horizon)


class TripleBarrier_numpy:
    def __init__(self, horizon=20, upper=0.0, lower=0.0, barrier_unit="fraction", vol_span=100.0):
        self.h, self.up, self.lo, self.unit, self.span = horizon, upper, lower, barrier_unit, vol_span

    def __call__(self, price):
        p = np.asarray(price, float)
        n, h = len(p), self.h
        out = np.full((n, 3), np.nan)
        sig = _sigma(p, self.span, h) if self.unit == "vol" else np.ones(n)
        for e in range(n - h):
            pe, s = p[e], sig[e]
            if not (np.isfinite(pe) and pe > 0) or not (np.isfinite(s) and s > 0):
                continue
            U, L = _levels(pe, self.up, self.lo, self.unit, s)
            res = None
            for k in range(1, h + 1):
                q = p[e + k]
                if not (np.isfinite(q) and q > 0):
                    break
                if q >= U:
                    res = (1.0, q, k); break
                if q <= L:
                    res = (-1.0, q, k); break
                if k == h:
                    res = (0.0, q, k)
            if res is not None:
                out[e + h] = (res[0], res[1] / pe - 1.0, res[2])
        return out
