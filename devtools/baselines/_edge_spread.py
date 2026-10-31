"""The EDGE bid-ask spread estimator (Ardia, Guidotti and Kroencke 2024),
written from the authors' pseudocode (github.com/eguidotti/bidask,
pseudocode/README.md).

``edge`` is the estimator on one sample of bars, with the pseudocode's
handling of missing values: each indicator and each return is left out where
a price it uses is missing, and each mean is over the values that are left.
``edge_rolling`` applies it to every trailing window of ``window`` bars, the
definition the authors' rolling implementation is tested against.
"""
import numpy as np


def _nanmean(x):
    x = np.asarray(x, dtype=float)
    keep = ~np.isnan(x)
    return x[keep].mean() if keep.any() else np.nan


def edge(open_, high, low, close, sign=False):
    if len(open_) < 3:
        return np.nan
    o, h, l, c = (np.log(np.asarray(a, dtype=float)) for a in (open_, high, low, close))
    m = (h + l) / 2.0
    h1, l1, c1, m1 = h[:-1], l[:-1], c[:-1], m[:-1]
    o, h, l, m = o[1:], h[1:], l[1:], m[1:]

    r1, r2, r3, r4, r5 = m - o, o - m1, m - c1, c1 - m1, o - c1

    def indicator(cond, *used):
        return np.where(np.any([np.isnan(u) for u in used], axis=0), np.nan, cond)

    tau = indicator((h != l) | (l != c1), h, l, c1)
    po1 = tau * indicator(o != h, o, h)
    po2 = tau * indicator(o != l, o, l)
    pc1 = tau * indicator(c1 != h1, c1, h1)
    pc2 = tau * indicator(c1 != l1, c1, l1)

    pt = _nanmean(tau)
    po = _nanmean(po1) + _nanmean(po2)
    pc = _nanmean(pc1) + _nanmean(pc2)
    if np.nansum(tau) < 2 or po == 0 or pc == 0:
        return np.nan

    d1 = r1 - _nanmean(r1) / pt * tau
    d3 = r3 - _nanmean(r3) / pt * tau
    d5 = r5 - _nanmean(r5) / pt * tau
    x1 = -4.0 / po * d1 * r2 + -4.0 / pc * d3 * r4
    x2 = -4.0 / po * d1 * r5 + -4.0 / pc * d5 * r4

    e1, e2 = _nanmean(x1), _nanmean(x2)
    v1 = _nanmean(x1 * x1) - e1 * e1
    v2 = _nanmean(x2 * x2) - e2 * e2
    vt = v1 + v2
    s2 = (v2 * e1 + v1 * e2) / vt if vt > 0 else (e1 + e2) / 2.0
    s = np.sqrt(np.abs(s2))
    return -s if sign and s2 < 0 else s


def edge_rolling(open_, high, low, close, window, sign=False, start_policy="strict"):
    """edge() on each trailing window of ``window`` bars; with
    start_policy="expanding", on all bars so far until the window is full."""
    arrays = [np.asarray(a, dtype=float) for a in (open_, high, low, close)]
    out = np.full(len(arrays[0]), np.nan)
    for t in range(len(out)):
        t0 = t + 1 - window
        if t0 < 0:
            if start_policy != "expanding":
                continue
            t0 = 0
        out[t] = edge(*(a[t0:t + 1] for a in arrays), sign=sign)
    return out
