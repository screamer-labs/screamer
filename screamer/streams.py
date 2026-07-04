"""Streaming combinators (merge, combine_latest, pace, dropna, filter, split).

Builds and runs C++ node graphs. dtype detection here chooses the int64 or
float64 key instantiation; the per-event work is all C++.
"""
import asyncio
import math

import numpy as np

from . import screamer_bindings as _b
from .dag import is_node, make_combinator_node

__all__ = [
    "Stream",
    "merge", "merge_iter",
    "combine_latest", "combine_latest_iter",
    "pace",
    "dropna", "dropna_iter",
    "filter", "filter_iter",
    "split",
    "resample", "resample_iter",
]


class Stream:
    """A sequence of values with an optional ordering index.

    values : np.ndarray, shape (T,) or (T, N).
    index  : np.ndarray of length T, or None. None means positional (row-number /
             arrival order) and stores nothing. The index is an ordering
             coordinate (timestamp, tick counter, ...), never a lookup key.
    """
    __slots__ = ("values", "index")

    def __init__(self, values, index=None):
        self.values = np.asarray(values)
        self.index = None if index is None else np.asarray(index)
        if self.index is not None and len(self.index) != len(self.values):
            raise ValueError("Stream: index and values must have the same length")

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        kind = "positional" if self.index is None else f"index={self.index!r}"
        return f"Stream({self.values!r}, {kind})"

    @classmethod
    def from_pandas(cls, obj):
        """Build a Stream from a pandas Series or DataFrame (data -> values,
        pandas index -> index). Note: a plain RangeIndex is kept as a numbered
        index, not converted to positional None."""
        return cls(obj.to_numpy(), np.asarray(obj.index))

    def to_pandas(self):
        """Return a pandas Series (1-D values) or DataFrame (2-D). A positional
        stream gets pandas' default RangeIndex."""
        import pandas as pd
        if self.values.ndim == 1:
            return pd.Series(self.values, index=self.index)
        return pd.DataFrame(self.values, index=self.index)


def _regime(inputs):
    """Classify a combinator's inputs: 'graph' if any is a Node, 'stream' if any
    is a Stream, else 'raw'."""
    if any(is_node(x) for x in inputs):
        return "graph"
    if any(isinstance(x, Stream) for x in inputs):
        return "stream"
    return "raw"


def _to_streams(inputs, index):
    """Normalize each input to a Stream. `index` is None (all positional) or a
    list aligned with inputs (per-stream index array or None)."""
    if index is not None and len(index) != len(inputs):
        raise ValueError("index list length must match the number of streams")
    out = []
    for i, x in enumerate(inputs):
        if isinstance(x, Stream):
            out.append(x)
        else:
            out.append(Stream(x, None if index is None else index[i]))
    return out


def _adapt(regime, values, index):
    """Shape a combinator result to match the input regime: Stream in -> Stream
    out; raw -> (values, index) with index None for positional."""
    if regime == "stream":
        return Stream(values, index)
    return values, index


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
    from the first event with NaN for not-yet-seen inputs. If ``func`` is given it
    is applied per row (``func(*row)``) and (keys, reduced) is returned instead.
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

    The inverse of merge: ``split(*merge(*series))`` reconstructs the inputs.
    ``n`` sets how many output streams to produce (default: max(sources)+1); pass it
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
        # graph form: select(stream, columns) - columns may be the 2nd
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


_RESAMPLE_AGGS = ("first", "last", "min", "max", "sum", "count", "mean", "ohlc")


class _ResampleAccum:
    """Single-pass O(1) NaN-ignore accumulator. Mirrors the C++ ResampleAccum."""
    __slots__ = ("count", "s", "mn", "mx", "first", "last", "has")

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.s = 0.0
        self.mn = 0.0
        self.mx = 0.0
        self.first = 0.0
        self.last = 0.0
        self.has = False

    def add(self, v):
        self.has = True
        if math.isnan(v):
            return
        if self.count == 0:
            self.mn = self.mx = self.first = self.last = v
        else:
            if v < self.mn:
                self.mn = v
            if v > self.mx:
                self.mx = v
            self.last = v
        self.s += v
        self.count += 1

    def emit(self, agg):
        nan = float("nan")
        c = self.count
        if agg == "first":
            return self.first if c else nan
        if agg == "last":
            return self.last if c else nan
        if agg == "min":
            return self.mn if c else nan
        if agg == "max":
            return self.mx if c else nan
        if agg == "sum":
            return self.s
        if agg == "count":
            return float(c)
        if agg == "mean":
            return self.s / c if c else nan
        # ohlc
        return [self.first if c else nan, self.mx if c else nan,
                self.mn if c else nan, self.last if c else nan]


def _resample_validate(width, count, agg, label):
    if (width is None) == (count is None):
        raise ValueError("resample: pass exactly one of width= or count=")
    # Positivity guard: width=0 would reach the engine's floordiv(_, 0) -> a hard
    # SIGFPE crash; count<1 never completes a bucket. Reject both up front (this
    # runs before the Node dispatch, so it guards the graph path too).
    if width is not None and int(width) <= 0:
        raise ValueError("resample: width must be positive")
    if count is not None and int(count) < 1:
        raise ValueError("resample: count must be >= 1")
    if agg not in _RESAMPLE_AGGS:
        raise ValueError(f"resample: agg must be one of {_RESAMPLE_AGGS}")
    if label not in ("left", "right"):
        raise ValueError('resample: label must be "left" or "right"')


def resample(keys, values=None, *, width=None, count=None, agg="last",
             origin=0, label="left"):
    """Causal windowed downsample of a width-1 (key, value) stream.

    Exactly one of `width` (fixed key-interval; buckets [origin+n*width,
    origin+(n+1)*width)) or `count` (fixed event-count). agg is one of
    first/last/min/max/sum/count/mean/ohlc (ohlc -> width-4). label "left"
    stamps the bucket start (by-key) or first key (by-count); "right" stamps the
    bucket end / last key. NaN values are ignored. Only non-empty buckets emit;
    the trailing partial bucket is emitted at end of input. Integer key-space.

    Graph form: resample(stream, ...) where stream is a Node.
    """
    _resample_validate(width, count, agg, label)
    if is_node(keys):
        return make_combinator_node(resample, (keys,), {
            "width": width, "count": count, "agg": agg,
            "origin": origin, "label": label})
    if values is None:
        raise ValueError("resample: values is required (eager form is "
                         "resample(keys, values, ...))")
    keys = np.asarray(keys)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("resample: expects a 1-D value stream (width-1)")

    out_keys, out_vals = [], []
    acc = _ResampleAccum()

    def flush_bucket(label_key):
        out_keys.append(label_key)
        out_vals.append(acc.emit(agg))

    if width is not None:
        w = int(width)
        o = int(origin)
        started = False
        bucket = 0
        cur_label = 0
        for k, v in zip(keys.tolist(), values.tolist()):
            k = int(k)
            nb = (k - o) // w          # Python // is floor division
            if not started:
                started = True
                bucket = nb
                acc.reset()
                cur_label = (o + nb * w) if label == "left" else (o + (nb + 1) * w)
            elif nb != bucket:
                if acc.has:
                    flush_bucket(cur_label)
                bucket = nb
                acc.reset()
                cur_label = (o + nb * w) if label == "left" else (o + (nb + 1) * w)
            acc.add(v)
        if acc.has:
            flush_bucket(cur_label)
    else:
        n = int(count)
        cib = 0
        first_key = 0
        last_key = 0
        for k, v in zip(keys.tolist(), values.tolist()):
            k = int(k)
            if cib == 0:
                first_key = k
            last_key = k
            acc.add(v)
            cib += 1
            if cib == n:
                flush_bucket(first_key if label == "left" else last_key)
                acc.reset()
                cib = 0
        if cib > 0:
            flush_bucket(first_key if label == "left" else last_key)

    ok = np.array(out_keys, dtype=np.int64)
    if agg == "ohlc":
        ov = (np.array(out_vals, dtype=np.float64).reshape(-1, 4)
              if out_vals else np.empty((0, 4), dtype=np.float64))
    else:
        ov = np.array(out_vals, dtype=np.float64)
    return ok, ov


def resample_iter(events, *, width=None, count=None, agg="last",
                  origin=0, label="left"):
    """Streaming resample over (key, value) tuples. Yields (label_key, value)."""
    _resample_validate(width, count, agg, label)
    acc = _ResampleAccum()
    if width is not None:
        w = int(width)
        o = int(origin)
        started = False
        bucket = 0
        cur_label = 0
        for k, v in events:
            k = int(k)
            nb = (k - o) // w
            if not started:
                started = True
                bucket = nb
                acc.reset()
                cur_label = (o + nb * w) if label == "left" else (o + (nb + 1) * w)
            elif nb != bucket:
                if acc.has:
                    yield cur_label, acc.emit(agg)
                bucket = nb
                acc.reset()
                cur_label = (o + nb * w) if label == "left" else (o + (nb + 1) * w)
            acc.add(float(v))
        if acc.has:
            yield cur_label, acc.emit(agg)
    else:
        n = int(count)
        cib = 0
        first_key = 0
        last_key = 0
        for k, v in events:
            k = int(k)
            if cib == 0:
                first_key = k
            last_key = k
            acc.add(float(v))
            cib += 1
            if cib == n:
                yield (first_key if label == "left" else last_key), acc.emit(agg)
                acc.reset()
                cib = 0
        if cib > 0:
            yield (first_key if label == "left" else last_key), acc.emit(agg)
