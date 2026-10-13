"""Offline supervised-learning helpers built on screamer's causal ops.

forecast_pairs builds a forecasting training set by delaying the features: each row
pairs features from the past with a target from their future. It delays X and passes y
through, so the pairing is causal and needs no future values of y. The target must
itself be causal (known as of its own index), typically a rolling trailing quantity.
These utilities are training-time only.

The graph is defined once in C++ (include/screamer/dag/forecast_pairs.h) and shared
with JavaScript's forecastPairs. Every regime (eager, lazy, a graph node in a
Pipeline, and so a live session) runs that graph; this module only adapts the inputs
and splits the output columns.
"""
from __future__ import annotations

import numpy as np

from .dag import _as_float64, _as_stream, _is_vi_pair, _setting, is_node, make_operator_node
from .streams import _is_lazy_stream

__all__ = ["forecast_pairs"]


def _horizon(count, duration):
    """count and duration as ints (None when absent), checked by the C++ rule.
    A string is a TypeError, as for every operator setting; a float is
    truncated, as Delay's duration is."""
    from . import screamer_bindings as _b
    if count is not None:
        count = int(_setting("forecast_pairs: count", count, True))
    if duration is not None:
        duration = int(_setting("forecast_pairs: duration", duration, True))
    _b._check_forecast_pairs_horizon(count, duration)
    return count, duration


def _graph(count, duration, dropna):
    from . import screamer_bindings as _b
    return _b._forecast_pairs_graph(count, duration, dropna)


def _split(out, x_values):
    """The output rows [X_shifted..., y] as (X_shifted, y); X_shifted has the
    rank of X."""
    wide = x_values.ndim == 2
    k = x_values.shape[1] if wide else 1
    if out.shape[0] == 0 and out.shape[1] != k + 1:   # no rows: X gives the width
        out = np.empty((0, k + 1))
    return (out[:, :k] if wide else out[:, 0]), out[:, k]


def forecast_pairs(X, y, *, count=None, duration=None, dropna=False):
    """Pair features with a target `count` events (or `duration` index-units) ahead.

    Returns (X_shifted, y). Row t holds the features from `count` events ago aligned
    with the target at t, so a model learns to predict `count` ahead. It delays X and
    passes y through unchanged. The first `count` rows of X_shifted are NaN (warmup);
    `dropna=True` drops any row whose shifted features or target is NaN. If you need to
    map rows back to time, keep your own index alongside X and y.

    Exactly one of `count` / `duration`. `count` is event-based and needs no index;
    `duration` is time-based (see Delay) and needs an index on X and y.

    Regimes
    -------
    Both count and duration mode support all three regimes:

    - Eager (arrays): call directly, returns ``(X_shifted, y)`` as numpy arrays.
    - Graph (Node inputs): returns a Node; wrap in a Pipeline to run it.
    - Lazy (generators of ``(value, index)`` events): returns a lazy iterator of
      ``((xs_val, y_val), index)`` events.

    Duration mode requires X and y to be ``(values, index)`` tuples in the eager
    regime; graph and lazy regimes carry the index implicitly through the pipeline.
    """
    count, duration = _horizon(count, duration)
    if not isinstance(dropna, (bool, np.bool_)):
        raise TypeError(f"forecast_pairs: dropna must be a bool, got {type(dropna).__name__}")
    dropna = bool(dropna)

    if is_node(X) or is_node(y):
        # Graph regime: one node that Pipeline compiles into the C++ graph.
        if not (is_node(X) and is_node(y)):
            raise TypeError("forecast_pairs: pass X and y both as Nodes to build a graph")
        return make_operator_node(forecast_pairs, (X, y),
                                  {"count": count, "duration": duration, "dropna": dropna})

    if _is_lazy_stream(X) or _is_lazy_stream(y):
        if not (_is_lazy_stream(X) and _is_lazy_stream(y)):
            raise TypeError(
                "forecast_pairs: cannot mix lazy iterator and concrete inputs; "
                "pass both X and y as generators or both as arrays")
        from . import screamer_bindings as _b
        return _b._LazyDriver(_graph(count, duration, dropna), [iter(X), iter(y)], 1)

    if duration is not None:
        # Duration is index-based: X and y come with their index.
        if not (isinstance(X, tuple) and isinstance(y, tuple)):
            raise TypeError("duration= mode needs X and y as (values, index) pairs")
        feeds = [_as_stream(X), _as_stream(y)]
        xv, yv = feeds[0][1], feeds[1][1]
    else:
        if _is_vi_pair(X) or _is_vi_pair(y):
            raise TypeError("count= mode takes X and y as values; (values, index) "
                            "pairs are for duration=")
        xv, yv = _as_float64(X), _as_float64(y)
        feeds = None
    from . import screamer_bindings as _b
    _b._check_forecast_pairs_inputs(
        duration is None, xv.ndim, len(xv), xv.shape[1] if xv.ndim > 1 else 1,
        yv.ndim, len(yv), yv.shape[1] if yv.ndim > 1 else 1)
    if feeds is None:
        feeds = [_as_stream(xv), _as_stream(yv)]
    (_, out), = _graph(count, duration, dropna).run_batch(feeds, False, "forecast_pairs")
    return _split(out, xv)
