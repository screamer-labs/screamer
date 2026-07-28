"""Offline supervised-learning helpers built on screamer's causal ops.

forecast_pairs builds a forecasting training set by delaying the features: each row
pairs features from the past with a target from their future. It delays X and passes y
through, so the pairing is causal and needs no future values of y. The target must
itself be causal (known as of its own index), typically a rolling trailing quantity.
These utilities are training-time only.

All three regimes (eager, graph, lazy) are supported for count mode via a Pipeline
of existing C++ nodes. Duration mode supports eager batch only; graph and lazy require
a CombineLatest emit="on_right" primitive that does not yet exist in the C++ core.
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
    """Duration-mode forecast_pairs: eager batch only.

    Shifts X's index by ``duration`` index units and aligns to y's clock via
    CombineLatest. Only rows at y-tick timestamps are returned (the y-clock
    selection is performed using a Python-level index set, since the existing
    C++ nodes do not yet provide an emit="on_right" mode for CombineLatest).

    Graph and lazy regimes are not supported until CombineLatest gains an
    emit="on_right" (or equivalent "emit only at the second stream's ticks")
    mode in the C++ core.
    """
    if is_node(X) or is_node(y):
        raise NotImplementedError(
            "forecast_pairs duration= mode does not support graph (Node) inputs. "
            "The y-clock selection requires a CombineLatest emit='on_right' primitive "
            "that does not yet exist in the C++ core. Use count= mode for graph/lazy regimes.")
    if _is_lazy_stream(X) or _is_lazy_stream(y):
        raise NotImplementedError(
            "forecast_pairs duration= mode does not support lazy iterator inputs. "
            "The y-clock selection requires a CombineLatest emit='on_right' primitive "
            "that does not yet exist in the C++ core. Use count= mode for graph/lazy regimes.")

    if not (isinstance(X, tuple) and isinstance(y, tuple)):
        raise TypeError("duration= mode needs X and y as (values, index) pairs")
    Xv, Xi = np.asarray(X[0], float), np.asarray(X[1])
    yv, yi = np.asarray(y[0], float), np.asarray(y[1])
    if Xv.ndim != 1 or yv.ndim != 1:
        raise ValueError("duration= mode supports 1-D X and y (one feature, one target)")

    # Delay X by duration (C++ node), then as-of join with y (C++ CombineLatest).
    Xsv, Xsi = Delay(int(duration))((Xv, Xi))
    combined, cidx = CombineLatest(emit="on_any")((Xsv, Xsi), (yv, yi))

    # Select only y-clock ticks (rows where the event index belongs to yi).
    # A Python-level set membership check is used because CombineLatest does not
    # yet have an emit="on_right" mode; no numpy masking functions are used here.
    yi_set = frozenset(cidx.dtype.type(v) for v in np.asarray(yi).tolist())
    at_ytick = np.array([cidx[i] in yi_set for i in range(len(cidx))], dtype=bool)

    Xs_out = combined[at_ytick, 0]
    y_out = combined[at_ytick, 1]

    if dropna_flag:
        valid = ~(np.isnan(Xs_out) | np.isnan(y_out))
        return Xs_out[valid], y_out[valid]
    return Xs_out, y_out


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
    count mode supports all three regimes:

    - Eager (arrays): call directly, returns ``(X_shifted, y)`` as numpy arrays.
    - Graph (Node inputs): returns a Node; wrap in a Pipeline to run it.
    - Lazy (generators of ``(value, index)`` events): returns a lazy iterator of
      ``((xs_val, y_val), index)`` events.

    duration mode supports eager batch only. Graph and lazy regimes raise
    NotImplementedError until CombineLatest gains an emit="on_right" mode.
    """
    if (count is None) == (duration is None):
        raise ValueError("pass exactly one of count= or duration=")
    if duration is not None:
        return _forecast_pairs_duration(X, y, duration, dropna)
    return _forecast_pairs_count(X, y, count, dropna)
