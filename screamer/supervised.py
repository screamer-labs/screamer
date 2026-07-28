"""Offline supervised-learning helpers built on screamer's causal ops.

forecast_pairs builds a forecasting training set by delaying the features: each row
pairs features from the past with a target from their future. It delays X and passes y
through, so the pairing is causal and needs no future values of y. The target must
itself be causal (known as of its own index), typically a rolling trailing quantity.
These utilities are training-time only.

All three regimes (eager, graph, lazy) are supported for both count and duration mode.
Duration mode uses Resample(clock=) to emit the last delayed X at each y-tick; this
is an as-of join on the index axis driven by the y clock.
"""
from __future__ import annotations

import numpy as np

from . import Lag, Delay, CombineLatest, Dropna, Select
from .dag import Input, Pipeline, is_node
from .streams import _is_lazy_stream

__all__ = ["forecast_pairs"]


def _build_count_pipeline(count, dropna_flag):
    """Build a reusable Pipeline for count-mode forecast_pairs.

    Inputs: 'X' (1-D values), 'y' (1-D values).
    Output: a single 2-column node [Xs, y], optionally filtered by Dropna.
    """
    X_in = Input("X")
    y_in = Input("y")
    Xs_node = Lag(int(count))(X_in)
    combined = CombineLatest()(Xs_node, y_in)
    if dropna_flag:
        combined = Dropna(how="any")(combined)
    return Pipeline([X_in, y_in], [combined])


def _forecast_pairs_count(X, y, count, dropna_flag):
    """Count-mode forecast_pairs: works in eager, graph, and lazy regimes."""
    if is_node(X) or is_node(y):
        # Graph regime: build the computation graph and return a Node.
        Xs_node = Lag(int(count))(X)
        combined = CombineLatest()(Xs_node, y)
        if dropna_flag:
            combined = Dropna(how="any")(combined)
        return combined

    if _is_lazy_stream(X) or _is_lazy_stream(y):
        # Lazy regime: build a Pipeline and run it lazily.
        if not (_is_lazy_stream(X) and _is_lazy_stream(y)):
            raise TypeError(
                "forecast_pairs: cannot mix lazy iterator and concrete inputs; "
                "pass both X and y as generators or both as arrays")
        X_in = Input("X")
        y_in = Input("y")
        Xs_node = Lag(int(count))(X_in)
        combined = CombineLatest()(Xs_node, y_in)
        if dropna_flag:
            combined = Dropna(how="any")(combined)
        dag = Pipeline([X_in, y_in], [combined])
        return dag(X, y)

    # Eager regime: run the Pipeline in batch mode.
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(X) != len(y):
        raise ValueError("X and y must share the same length (time axis)")

    if X.ndim > 1:
        # Multi-column X: build an all-C++ Pipeline (CombineLatest + Dropna)
        # so no data crosses the Python/C++ boundary per event.
        # Lag handles 2-D natively; each feature column is Lag'd by count, then
        # all feature columns and y are combined via CombineLatest and optionally
        # filtered by Dropna. The output is a single (N+1)-wide array; the X
        # columns are 0..N-1 and y is column N. Column extraction from the final
        # batch result is shape-only I/O, not per-event Python logic.
        N = X.shape[1]
        col_inputs = [Input(f"X{i}") for i in range(N)]
        y_in = Input("y")
        lagged = [Lag(int(count))(c) for c in col_inputs]
        combined = CombineLatest()(*lagged, y_in)
        if dropna_flag:
            combined = Dropna(how="any")(combined)
        dag = Pipeline(col_inputs + [y_in], [combined])
        col_feeds = [X[:, i] for i in range(N)]
        result, _ = dag(*col_feeds, y)
        # result is (rows, N+1): cols 0..N-1 are lagged features, col N is y.
        return result[:, :N], result[:, N]

    # 1-D X: build and run the Pipeline.
    dag = _build_count_pipeline(count, dropna_flag)
    result, _ = dag(X, y)
    return result[:, 0], result[:, 1]


def _forecast_pairs_duration(X, y, duration, dropna_flag):
    """Duration-mode forecast_pairs: all regimes (eager, graph, lazy).

    Shifts X's index by ``duration`` index units via Delay, then resamples at
    each y-tick using Resample(clock=y, agg='last', fill='carry'). CombineLatest
    pairs the resampled Xs with the corresponding y value at each clock tick.
    Dropna filters rows where either Xs or y is NaN when requested.

    Works in all three regimes by building a Pipeline of C++ nodes; no
    Python-level data processing occurs.
    """
    from .streams import Resample

    def _build_dag(X_in, y_in):
        """Build the duration-mode pipeline on the given input nodes."""
        Xs_node = Resample(Delay(int(duration))(X_in), y_in,
                           clock=True, agg='last', fill='carry')
        combined = CombineLatest()(Xs_node, y_in)
        if dropna_flag:
            return Dropna(how="any")(combined)
        return combined

    if is_node(X) or is_node(y):
        # Graph regime: wire directly on the provided Node handles.
        return _build_dag(X, y)

    if _is_lazy_stream(X) or _is_lazy_stream(y):
        # Lazy regime: run a Pipeline over the two iterator feeds.
        X_in = Input("X")
        y_in = Input("y")
        dag = Pipeline([X_in, y_in], [_build_dag(X_in, y_in)])
        return dag(iter(X), iter(y))

    # Eager regime: X and y must be (values, index) tuples (duration is index-based).
    if not (isinstance(X, tuple) and isinstance(y, tuple)):
        raise TypeError("duration= mode needs X and y as (values, index) pairs")

    X_in = Input("X")
    y_in = Input("y")
    dag = Pipeline([X_in, y_in], [_build_dag(X_in, y_in)])
    result, _ = dag(X, y)
    return result[:, 0], result[:, 1]


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
    if (count is None) == (duration is None):
        raise ValueError("pass exactly one of count= or duration=")
    if duration is not None:
        return _forecast_pairs_duration(X, y, duration, dropna)
    return _forecast_pairs_count(X, y, count, dropna)
