"""Triple-barrier labels on OHLC bars written as a plain loop over entries."""
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
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(price[1:] / price[:-1])
    r[~((price[1:] > 0) & (price[:-1] > 0))] = np.nan
    r[~np.isfinite(r)] = np.nan
    s = pd.Series(np.r_[np.nan, r]).ewm(span=vol_span, ignore_na=True).std().to_numpy()
    return s * np.sqrt(horizon)


def _valid(x):
    return np.isfinite(x) and x > 0


class TripleBarrierOHLC_numpy:
    def __init__(self, horizon=20, upper=0.0, lower=0.0, barrier_unit="fraction", vol_span=100.0):
        self.h, self.up, self.lo, self.unit, self.span = horizon, upper, lower, barrier_unit, vol_span

    def __call__(self, open, high, low, close):
        o, hi, lo, c = (np.asarray(a, float) for a in (open, high, low, close))
        n, h = len(c), self.h
        out = np.full((n, 3), np.nan)
        sig = _sigma(c, self.span, h) if self.unit == "vol" else np.ones(n)
        for e in range(n - h):
            pe, s = c[e], sig[e]
            if not _valid(pe) or not (np.isfinite(s) and s > 0):
                continue
            U, L = _levels(pe, self.up, self.lo, self.unit, s)
            res = None
            for k in range(1, h + 1):
                b = e + k
                if not all(_valid(x) for x in (o[b], hi[b], lo[b], c[b])) or lo[b] > hi[b]:
                    break                                   # missing or inconsistent bar
                if o[b] >= U:
                    res = (1.0, o[b], k); break
                if o[b] <= L:
                    res = (-1.0, o[b], k); break
                if hi[b] >= U and lo[b] <= L:
                    break                                   # order inside the bar unknown
                if hi[b] >= U:
                    res = (1.0, np.clip(U, lo[b], hi[b]), k); break
                if lo[b] <= L:
                    res = (-1.0, np.clip(L, lo[b], hi[b]), k); break
                if k == h:
                    res = (0.0, c[b], k)
            if res is not None:
                out[e + h] = (res[0], res[1] / pe - 1.0, res[2])
        return out
