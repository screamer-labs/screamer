"""Offline supervised-learning helpers built on screamer's causal ops.

forecast_pairs builds a forecasting training set. A training row holds the features
known at time t next to what happened between t and t + h. Live, the same row exists
with the target still unknown (NaN), so the features are computed by the same code at
the same timestamps in research and live, and a model learns "features now, outcome
next". The target must itself be causal (known as of its own index), typically a
rolling trailing quantity such as ``LogReturn(h)``; pairing moves it back to the
features' row, so no future value is computed. These utilities are training-time only.

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


def _index(index):
    """The index setting as a bool (True for "features"), checked by the C++ rule."""
    from . import screamer_bindings as _b
    if not isinstance(index, str):
        raise TypeError(f"forecast_pairs: index must be a string, got {type(index).__name__}")
    return _b._forecast_pairs_index(index)


def _graph(count, duration, dropna, features):
    from . import screamer_bindings as _b
    return _b._forecast_pairs_graph(count, duration, dropna, features)


def _split(out, x_values):
    """The output rows [X..., y] as (X, y); the X part has the rank of X."""
    wide = x_values.ndim == 2
    k = x_values.shape[1] if wide else 1
    if out.shape[0] == 0 and out.shape[1] != k + 1:   # no rows: X gives the width
        out = np.empty((0, k + 1))
    return (out[:, :k] if wide else out[:, 0]), out[:, k]


def forecast_pairs(X, y, *, count=None, duration=None, index="features", dropna=False):
    """Pair features with the target over a horizon after them.

    Returns ``(X_paired, y_paired)``, a training set for a model that predicts the
    target from the features. ``y`` is a causal series: its value at step t is known
    at t, so a target over a horizon h is stamped at the end of that horizon (for
    example ``LogReturn(h)(price)`` at t is the return over (t - h, t]).

    The horizon is exactly one of ``count`` (a number of events; no index needed) or
    ``duration`` (index units, see Delay; X and y carry an index).

    ``index`` chooses whose timestamp a row carries:

    - ``"features"`` (the default): row t holds ``X[t]`` and the target over
      (t, t + h]. This is the row a live model sees, with the target filled in. The
      last rows, whose target is not known yet, have a NaN target. In count mode
      the target is ``y[t + count]``. In duration mode there is one row per X event,
      holding the last y at or before ``t + duration``, NaN when y ends before it.
    - ``"target"``: row t holds ``X[t - h]`` and ``y[t]``, stamped when the target is
      known. In count mode the first ``count`` feature rows are NaN. In duration mode
      there is one row per y event, holding the last X at or before
      ``t - duration``.

    In count mode both settings hold the same pairs; with ``dropna=True``, which
    drops every row that holds a NaN, they return identical arrays. If you need to
    map rows back to time, keep your own index alongside X and y.

    Regimes
    -------
    - Eager (arrays): returns ``(X_paired, y_paired)`` as numpy arrays; X_paired has
      the rank of X.
    - Graph (Node inputs): returns a Node; wrap it in a Pipeline to run it, also as
      a live session.
    - Lazy (generators of ``(value, index)`` events): returns a lazy iterator of
      ``((x..., y), index)`` events.

    Streaming with ``index="features"``, the row for t is emitted once its target
    is known (``count`` later events have arrived, or y has passed ``t + duration``),
    still stamped t, so stamps stay in order. At the end of input the rows still
    waiting are emitted with a NaN target. Memory is the rows waiting for their
    target, not the dataset.

    Duration mode takes X and y as ``(values, index)`` tuples in the eager regime;
    graph and lazy regimes carry the index with each event.
    """
    features = _index(index)
    count, duration = _horizon(count, duration)
    if not isinstance(dropna, (bool, np.bool_)):
        raise TypeError(f"forecast_pairs: dropna must be a bool, got {type(dropna).__name__}")
    dropna = bool(dropna)

    if is_node(X) or is_node(y):
        # Graph regime: one node that Pipeline compiles into the C++ graph.
        if not (is_node(X) and is_node(y)):
            raise TypeError("forecast_pairs: pass X and y both as Nodes to build a graph")
        return make_operator_node(forecast_pairs, (X, y),
                                  {"count": count, "duration": duration, "index": index,
                                   "dropna": dropna})

    if _is_lazy_stream(X) or _is_lazy_stream(y):
        if not (_is_lazy_stream(X) and _is_lazy_stream(y)):
            raise TypeError(
                "forecast_pairs: cannot mix lazy iterator and concrete inputs; "
                "pass both X and y as generators or both as arrays")
        from . import screamer_bindings as _b
        return _b._LazyDriver(_graph(count, duration, dropna, features), [iter(X), iter(y)], 1)

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
    (_, out), = _graph(count, duration, dropna, features).run_batch(feeds, False, "forecast_pairs")
    return _split(out, xv)
