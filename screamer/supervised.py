"""Offline supervised-learning helpers built on screamer's causal ops.

forecast_pairs builds a forecasting training set by delaying the features: each row
pairs features from the past with a target from their future. It delays X and passes y
through, so the pairing is causal and needs no future values of y. The target must
itself be causal (known as of its own index), typically a rolling trailing quantity.
These utilities are training-time only.
"""
from __future__ import annotations

import numpy as np

from . import Lag, Delay, CombineLatest

__all__ = ["forecast_pairs"]


def _leading_nan_mask(a):
    """True where a row is fully finite (a is 1-D or 2-D, per-row over columns)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return np.isfinite(a)
    return np.isfinite(a).all(axis=tuple(range(1, a.ndim)))


def _forecast_pairs_duration(X, y, duration, dropna):
    """Duration-mode helper: shift X index by duration, align to y clock via CombineLatest."""
    if not (isinstance(X, tuple) and isinstance(y, tuple)):
        raise TypeError("duration= mode needs X and y as (values, index) pairs")
    Xv, Xi = np.asarray(X[0], float), np.asarray(X[1])
    yv, yi = np.asarray(y[0], float), np.asarray(y[1])
    if Xv.ndim != 1 or yv.ndim != 1:
        raise ValueError("duration= mode supports 1-D X and y (one feature, one target)")
    Xsv, Xsi = Delay(int(duration))(Xv, Xi)
    combined, cidx = CombineLatest(emit="on_any")((Xsv, Xsi), (yv, yi))
    keep_clock = np.isin(cidx, np.asarray(yi, dtype=cidx.dtype))
    Xs = combined[keep_clock, 0]
    ys = combined[keep_clock, 1]
    if dropna:
        m = np.isfinite(Xs) & np.isfinite(ys)
        return Xs[m], ys[m]
    return Xs, ys


def forecast_pairs(X, y, *, count=None, duration=None, dropna=False):
    """Pair features with a target `count` events (or `duration` index-units) ahead.

    Returns (X_shifted, y). Row t holds the features from `count` events ago aligned
    with the target at t, so a model learns to predict `count` ahead. It delays X and
    passes y through unchanged. The first `count` rows of X_shifted are NaN (warmup);
    `dropna=True` drops any row whose shifted features or target is NaN. If you need to
    map rows back to time, keep your own index alongside X and y.

    Exactly one of `count` / `duration`. `count` is event-based and needs no index;
    `duration` is time-based (see Delay) and needs an index on X and y.
    """
    if (count is None) == (duration is None):
        raise ValueError("pass exactly one of count= or duration=")
    if duration is not None:
        return _forecast_pairs_duration(X, y, duration, dropna)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(X) != len(y):
        raise ValueError("X and y must share the same length (time axis)")
    Xs = np.asarray(Lag(int(count))(X), dtype=float)
    if dropna:
        keep = _leading_nan_mask(Xs) & _leading_nan_mask(y)   # drop rows where either is NaN
        return Xs[keep], y[keep]
    return Xs, y
