"""Streaming combinators (merge, combine_latest, pace, dropna, filter, split).

Builds and runs C++ node graphs. dtype detection here chooses the int64 or
float64 key instantiation; the per-event work is all C++.
"""
import asyncio

import numpy as np

from . import screamer_bindings as _b
from .dag import is_node, make_combinator_node

__all__ = [
    "merge", "merge_iter",
    "combine_latest", "combine_latest_iter",
    "pace",
    "dropna", "dropna_iter",
    "filter", "filter_iter",
    "split",
]


def _run_chain(functors, values, keys=None, return_keys=False):
    """Run source -> functors[0] -> ... -> collector in batch.

    Returns values array by default. Pass return_keys=True to get
    (out_keys, out_values) instead.

    keys=None uses the row number (int64) as the ordering key.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    n = values.shape[0]
    functors = list(functors)

    if keys is None:
        keys = np.arange(n, dtype=np.int64)
        kind = "i64"
    else:
        kind, keys = _key_dtype_kind(keys)

    fn = _b._run_chain_f64 if kind == "f64" else _b._run_chain_i64
    result = fn(functors, keys, values, return_keys)
    return result  # tuple (keys, values) if return_keys else values array


def _key_dtype_kind(keys):
    """Return ('f64' | 'i64', normalized_keys) for a single key array."""
    keys = np.asarray(keys)
    if np.issubdtype(keys.dtype, np.floating):
        return "f64", np.ascontiguousarray(keys, dtype=np.float64)
    if keys.dtype.kind == "M":
        keys = keys.view("int64")
    return "i64", np.ascontiguousarray(keys, dtype=np.int64)


def _normalize_series(series, who):
    """Normalize and validate a sequence of (keys, values) series.

    Returns (kind, norm_keys, norm_vals) where kind is 'i64' or 'f64'.
    Raises ValueError if series is empty, TypeError if key types differ.
    """
    if not series:
        raise ValueError(f"{who}: needs at least one series")
    kinds = set()
    norm_keys, norm_vals = [], []
    for keys, values in series:
        kind, k = _key_dtype_kind(keys)
        kinds.add(kind)
        norm_keys.append(k)
        norm_vals.append(np.ascontiguousarray(values, dtype=np.float64))
    if len(kinds) != 1:
        raise TypeError(f"{who}: all series must share one key type (all int/datetime or all float)")
    return kinds.pop(), norm_keys, norm_vals


def merge(*series):
    """Merge N (keys, values) series into one key-sorted (keys, values, sources).

    Each series must be individually sorted by key. `sources[i]` is the index
    of the series that emitted event i. Ties break by series order.
    """
    if any(is_node(s) for s in series):
        return make_combinator_node(merge, series, {})
    kind, norm_keys, norm_vals = _normalize_series(series, "merge")
    fn = _b._merge_f64 if kind == "f64" else _b._merge_i64
    return fn(norm_keys, norm_vals)


def _make_merge_puller(series):
    kind, norm_keys, norm_vals = _normalize_series(series, "merge_iter")
    cls = _b._MergePuller_f64 if kind == "f64" else _b._MergePuller_i64
    return cls(norm_keys, norm_vals)


def merge_iter(*series):
    """Yield (key, value, source) events in key order, pulled one at a time."""
    puller = _make_merge_puller(series)
    while True:
        event = puller.next()
        if event is None:
            return
        yield event


def _merge_events(*series):
    """Return a list of (key, value, source) tuples from merge_iter (test helper)."""
    return list(merge_iter(*series))


def combine_latest(*series, emit="when_all", func=None):
    """As-of latest-value join of N (keys, values) series.

    Emits an aligned row whenever any input advances, carrying each input's most
    recent value (forward-fill). Returns (keys, aligned) where aligned is (M, N);
    aligned[:, j] is series j's latest value at each emitted key. emit="when_all"
    (default) suppresses output until every input is warm; emit="on_any" emits
    from the first event with NaN for not-yet-seen inputs. If `func` is given it is
    applied per row (func(*row)) and (keys, reduced) is returned instead.
    """
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    if any(is_node(s) for s in series):
        if func is not None:
            raise ValueError(
                "combine_latest(func=...) is not supported in a DAG graph "
                "(graph ops are C++-only); apply a C++ functor to the aligned "
                "output instead, e.g. Sub()(combine_latest(a, b))")
        return make_combinator_node(combine_latest, series, {"emit": emit, "func": None})
    kind, norm_keys, norm_vals = _normalize_series(series, "combine_latest")
    fn = _b._combine_latest_f64 if kind == "f64" else _b._combine_latest_i64
    keys, aligned = fn(norm_keys, norm_vals, emit == "when_all")
    if func is None:
        return keys, aligned
    reduced = np.array([func(*row) for row in aligned], dtype=np.float64)
    return keys, reduced


def combine_latest_iter(*series, emit="when_all"):
    """Yield (key, (v0, v1, ...)) aligned rows one at a time (streaming form)."""
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    kind, norm_keys, norm_vals = _normalize_series(series, "combine_latest")
    cls = _b._CombineLatestPuller_f64 if kind == "f64" else _b._CombineLatestPuller_i64
    puller = cls(norm_keys, norm_vals, emit == "when_all")
    while True:
        event = puller.next()
        if event is None:
            return
        yield event


async def pace(*series, speed=1.0, sleep=None):
    """Replay merged series as an async event stream paced by key-deltas.

    Yields (key, value, source) in key order. Between consecutive events it
    awaits `sleep(key_delta / speed)` so wall-clock spacing tracks the key
    spacing. speed=inf disables pacing (backtest at max speed). Pacing never
    changes values or order. `sleep` is injectable for testing; defaults to
    asyncio.sleep. Requires a metric (subtractable) key.
    """
    if speed <= 0:
        raise ValueError("pace: speed must be positive (or float('inf') for no pacing)")
    if sleep is None:
        sleep = asyncio.sleep
    infinite = speed == float("inf")
    prev_key = None
    for key, value, source in merge_iter(*series):
        if not infinite and prev_key is not None:
            delta = key - prev_key
            wait = delta / speed
            if wait > 0:
                await sleep(wait)
        prev_key = key
        yield key, value, source


def dropna(keys, values=None, how="any"):
    """Drop events whose value is NaN. `values` may be 1-D (M,) or 2-D (M, N).

    how="any" (default) drops a row if any component is NaN; how="all" only if
    all are. Causal and cardinality-changing: batch and streaming drop the same
    rows. Returns (keys, values) restricted to the surviving rows. Surviving
    values are returned as float64 (values are cast for the NaN test).
    """
    if how not in ("any", "all"):
        raise ValueError('dropna: how must be "any" or "all"')
    if is_node(keys):
        return make_combinator_node(dropna, (keys,), {"how": how})
    keys = np.asarray(keys)
    values = np.asarray(values, dtype=np.float64)
    nan = np.isnan(values)
    if values.ndim == 1:
        mask = ~nan
    else:
        mask = ~(nan.any(axis=1) if how == "any" else nan.all(axis=1))
    return keys[mask], values[mask]


def filter(keys, values=None, predicate=None):
    """Keep events where predicate(row) is truthy.

    row is a scalar for a 1-D value stream, a 1-D array for a 2-D aligned stream.
    predicate is a Python callable (per-event), so heavy filtering should prefer
    dropna or numpy masks; filter is the general escape hatch.
    """
    if is_node(keys):
        raise ValueError(
            "filter is not supported as a DAG graph node: the graph engine has "
            "no Python predicates (no lambda). Use dropna for NaN removal.")
    keys = np.asarray(keys)
    values = np.asarray(values)
    mask = np.fromiter((bool(predicate(row)) for row in values),
                       dtype=bool, count=len(values))
    return keys[mask], values[mask]


def dropna_iter(events, how="any"):
    """Streaming dropna over (key, value) tuples. value may be scalar or sequence."""
    if how not in ("any", "all"):
        raise ValueError('dropna_iter: how must be "any" or "all"')
    for key, value in events:
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        nan = np.isnan(arr)
        drop = nan.any() if how == "any" else nan.all()
        if not drop:
            yield key, value


def filter_iter(events, predicate):
    """Streaming filter over (key, value) tuples."""
    for key, value in events:
        if predicate(value):
            yield key, value


def split(keys, values, sources, n=None):
    """Partition a merged tagged stream back into per-source (keys, values).

    The inverse of merge: split(*merge(*series)) reconstructs the inputs. `n`
    sets how many output streams to produce (default: max(sources)+1); pass it
    explicitly to include sources that emitted nothing.
    """
    keys = np.asarray(keys)
    values = np.asarray(values)
    sources = np.asarray(sources)
    if n is None:
        n = int(sources.max()) + 1 if sources.size else 0
    elif sources.size and n <= int(sources.max()):
        raise ValueError(
            f"split: n={n} is too small for sources up to {int(sources.max())}; "
            "events would be dropped")
    return [(keys[sources == i], values[sources == i]) for i in range(n)]


def _normalize_columns(columns):
    """Validate columns (int or non-negative int sequence) -> (list_of_ints, scalar).

    Returns (cols, is_scalar). is_scalar is True when a bare int was given (the
    eager result is then 1-D). Negative indices are rejected explicitly.
    """
    scalar = np.ndim(columns) == 0
    cols = [int(columns)] if scalar else [int(c) for c in columns]
    for c in cols:
        if c < 0:
            raise ValueError(f"select: column index must be non-negative, got {c}")
    return cols, scalar


def select(keys, values=None, columns=None):
    """Pick column(s) from a wide (M, N) value stream.

    columns is an int (result is 1-D) or a sequence of ints (result is 2-D with
    those columns in order). Keys and row count are unchanged (shape op, not
    cardinality). Indices must be in range and non-negative.

    Graph form: select(stream, columns) where stream is a Node.
    """
    if is_node(keys):
        # graph form: select(stream, columns) — columns may be the 2nd
        # positional (the `values` slot) or the `columns` keyword.
        cols = values if columns is None else columns
        if cols is None:
            raise ValueError("select: columns is required")
        return make_combinator_node(select, (keys,), {"columns": cols})
    if values is None:
        raise ValueError("select: values is required (eager form is "
                         "select(keys, values, columns))")
    if columns is None:
        raise ValueError("select: columns is required")
    keys = np.asarray(keys)
    values = np.asarray(values, dtype=np.float64)
    cols, scalar = _normalize_columns(columns)
    if values.ndim == 1:
        width = 1
    else:
        width = values.shape[1]
    for c in cols:
        if c >= width:
            raise ValueError(
                f"select: column {c} out of range for width {width}")
    if values.ndim == 1:
        # width 1: only column 0 is valid; result mirrors input
        picked = values if scalar else values.reshape(-1, 1)
    else:
        picked = values[:, cols[0]] if scalar else values[:, cols]
    return keys, picked


def select_iter(events, columns):
    """Streaming select over (key, value) tuples. value is scalar or sequence."""
    cols, scalar = _normalize_columns(columns)
    for key, value in events:
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        for c in cols:
            if c >= arr.size:
                raise ValueError(
                    f"select_iter: column {c} out of range for width {arr.size}")
        if scalar:
            yield key, float(arr[cols[0]])
        else:
            yield key, [float(arr[c]) for c in cols]
