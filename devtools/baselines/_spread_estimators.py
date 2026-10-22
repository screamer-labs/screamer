"""Corwin-Schultz (2012) and Abdi-Ranaldo (2017) spread estimators, written
from the papers' formulas.

Both estimate the spread of each pair of consecutive bars and average the
pair estimates over a trailing window. The estimate of the pair (t-1, t) is
reported at bar t. A bar with a NaN high, low or close is left out, so the bars
on either side of it form a pair.
"""
import numpy as np

from ._windowing import trailing


def _usable(*arrays):
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    keep = ~np.any([np.isnan(a) for a in arrays], axis=0)
    return keep, [a[keep] for a in arrays]


def _windowed(pairs, n_bars, window, start_policy, finish):
    """Mean of the pair values over trailing windows, as bar-indexed output."""
    out = np.full(n_bars, np.nan)
    for j, w in enumerate(trailing(pairs, window, start_policy)):
        if w is not None:
            out[j + 1] = finish(np.sum(w) / (window if start_policy == "zero" else len(w)))
    return out


def corwin_schultz_pairs(high, low, close, adjust_overnight=True):
    """Two-bar Corwin-Schultz spreads, negative estimates set to 0."""
    hp, lp, cp = high[:-1], low[:-1], close[:-1]
    h, l = high[1:].copy(), low[1:].copy()
    if adjust_overnight:
        # The overnight adjustment: shift bar t's high and low by the gap from close t-1.
        up = l > cp
        gap = l - cp
        h[up] -= gap[up]
        l[up] -= gap[up]
        down = ~up & (h < cp)
        gap = cp - h
        h[down] += gap[down]
        l[down] += gap[down]
    beta = np.log(hp / lp) ** 2 + np.log(h / l) ** 2
    gamma = np.log(np.maximum(hp, h) / np.minimum(lp, l)) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return np.maximum(s, 0.0)


def abdi_ranaldo_pairs(high, low, close):
    """Two-bar products 4 (c_{t-1} - eta_{t-1}) (c_{t-1} - eta_t)."""
    c = np.log(close)
    eta = (np.log(high) + np.log(low)) / 2.0
    return 4.0 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])


def corwin_schultz(high, low, close, window, adjust_overnight=True, start_policy="strict"):
    keep, (h, l, c) = _usable(high, low, close)
    out = np.full(len(keep), np.nan)
    if len(h) > 1:
        pairs = corwin_schultz_pairs(h, l, c, adjust_overnight)
        out[keep] = _windowed(pairs, len(h), window, start_policy, lambda m: m)
    return out


def abdi_ranaldo(high, low, close, window, start_policy="strict"):
    keep, (h, l, c) = _usable(high, low, close)
    out = np.full(len(keep), np.nan)
    if len(h) > 1:
        pairs = abdi_ranaldo_pairs(h, l, c)
        out[keep] = _windowed(pairs, len(h), window, start_policy,
                              lambda m: np.sqrt(max(m, 0.0)))
    return out
