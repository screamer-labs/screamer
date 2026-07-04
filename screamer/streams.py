"""Streaming operators (merge, combine_latest, pace, dropna, filter, split).

Builds and runs C++ node graphs. dtype detection here chooses the int64 or
float64 index-type instantiation; the per-event work is all C++.
"""
import asyncio
import math

import numpy as np

from . import screamer_bindings as _b
from .dag import is_node, make_operator_node

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
    """Classify a stream operator's inputs: 'graph' if any is a Node, 'stream' if any
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
    """Shape a stream operator result to match the input regime: Stream in -> Stream
    out; raw -> (values, index) with index None for positional."""
    if regime == "stream":
        return Stream(values, index)
    return values, index


def _run_chain(functors, values, index=None, return_index=False):
    """Run source -> functors[0] -> ... -> collector in batch.

    Returns values array by default. Pass return_index=True to get
    (out_index, out_values) instead.

    index=None uses the row number (int64) as the ordering index.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    n = values.shape[0]
    functors = list(functors)

    if index is None:
        index = np.arange(n, dtype=np.int64)
        kind = "i64"
    else:
        kind, index = _index_dtype_kind(index)

    fn = _b._run_chain_f64 if kind == "f64" else _b._run_chain_i64
    result = fn(functors, index, values, return_index)
    return result  # tuple (index, values) if return_index else values array


def _index_dtype_kind(index):
    """Return ('f64' | 'i64', normalized_index) for a single index array."""
    index = np.asarray(index)
    if np.issubdtype(index.dtype, np.floating):
        return "f64", np.ascontiguousarray(index, dtype=np.float64)
    if index.dtype.kind == "M":
        index = index.view("int64")
    return "i64", np.ascontiguousarray(index, dtype=np.int64)


def _normalize_streams(streams, who):
    """Normalize and validate a sequence of (index, values) streams.

    Returns (kind, norm_index, norm_vals) where kind is 'i64' or 'f64'.
    Raises ValueError if streams is empty, TypeError if index types differ.
    """
    if not streams:
        raise ValueError(f"{who}: needs at least one stream")
    kinds = set()
    norm_index, norm_vals = [], []
    for idx, values in streams:
        kind, k = _index_dtype_kind(idx)
        kinds.add(kind)
        norm_index.append(k)
        norm_vals.append(np.ascontiguousarray(values, dtype=np.float64))
    if len(kinds) != 1:
        raise TypeError(f"{who}: all streams must share one index type (all int/datetime or all float)")
    return kinds.pop(), norm_index, norm_vals


def _collapse_last_per_index(index, values):
    """Keep the last row of each run of equal index (one row per distinct index).
    index must be non-decreasing (the aligner emits in index order)."""
    n = len(index)
    if n == 0:
        return index, values
    keep = np.empty(n, dtype=bool)
    keep[:-1] = index[:-1] != index[1:]
    keep[-1] = True
    return index[keep], values[keep]


def _streams_to_indexed(streams, who):
    """(kind, index_list, vals_list, positional). Uniform positional or indexed;
    no-index requires equal length; mixing positional and indexed raises."""
    indexed = [s.index is not None for s in streams]
    if any(indexed) and not all(indexed):
        raise ValueError(
            f"{who}: cannot align positional and indexed streams; give every "
            "stream an index, or none")
    vals = [np.ascontiguousarray(s.values, dtype=np.float64) for s in streams]
    if not any(indexed):
        lens = {len(s) for s in streams}
        if len(lens) != 1:
            raise ValueError(
                f"{who}: streams have no index, so they are assumed aligned - "
                "lengths must match, or provide an index to align different clocks")
        idx = [np.arange(len(streams[0]), dtype=np.int64) for _ in streams]
        return "i64", idx, vals, True
    kind, idx, _ = _normalize_streams([(s.index, s.values) for s in streams], who)
    return kind, idx, vals, False


def _merge_to_indexed(values, index, who):
    """Prepare merge inputs for the C++ backend.

    Positional (index=None or all-None list): uses per-stream row-number index
    (no equal-length check - unlike combine_latest). Indexed: uses the given
    index arrays. Mixing positional with indexed raises.

    Returns (kind, idx_list, vals_list, positional).
    """
    n = len(values)
    if n == 0:
        raise ValueError(f"{who}: needs at least one stream")
    per_stream_idx = [None] * n if index is None else list(index)
    if len(per_stream_idx) != n:
        raise ValueError(f"{who}: index list length must match number of streams")
    has_idx = [idx is not None for idx in per_stream_idx]
    if any(has_idx) and not all(has_idx):
        raise ValueError(
            f"{who}: cannot mix positional and indexed streams; give every "
            "stream an index, or none")
    vals = [np.ascontiguousarray(v, dtype=np.float64) for v in values]
    if not any(has_idx):
        # Positional: use row-number per stream (no equal-length check for merge)
        idx_list = [np.arange(len(v), dtype=np.int64) for v in vals]
        return "i64", idx_list, vals, True
    kind, norm_index, _ = _normalize_streams(list(zip(per_stream_idx, values)), who)
    return kind, norm_index, vals, False


def merge(*values, index=None):
    """Merge N value streams into one index-sorted (values, sources, index).

    Each stream must be individually sorted by its index. sources[i] is the
    source-stream index of event i. Ties break by stream order.

    Positional (index=None): uses row-number per stream; returned index is None.
    Unlike combine_latest, positional merge does NOT require equal lengths.
    Indexed (index=[idx0, idx1, ...]): merges by the given index arrays; returns
    the merged index array. Mixing positional and indexed raises ValueError.
    A Node input builds a graph node.
    """
    if any(is_node(v) for v in values):
        raise ValueError(
            "merge is not supported as a DAG graph node (it is input routing; "
            "feed streams to a Dag directly)")
    kind, idx_list, vals_list, positional = _merge_to_indexed(values, index, "merge")
    fn = _b._merge_f64 if kind == "f64" else _b._merge_i64
    merged_index, merged_vals, sources = fn(idx_list, vals_list)
    return merged_vals, sources, (None if positional else merged_index)


def merge_iter(*values, index=None):
    """Yield (value, index, source) events in index order.

    Positional inputs yield (value, None, source). Indexed inputs yield
    (value, index_value, source). See merge() for the positional/indexed rules.
    """
    kind, idx_list, vals_list, positional = _merge_to_indexed(values, index, "merge_iter")
    cls = _b._MergePuller_f64 if kind == "f64" else _b._MergePuller_i64
    puller = cls(idx_list, vals_list)
    while True:
        event = puller.next()
        if event is None:
            return
        ev_index, ev_val, ev_source = event
        yield ev_val, (None if positional else ev_index), ev_source


def combine_latest(*values, index=None, emit="when_all", func=None):
    """As-of latest-value join of N streams: one row per distinct index (same-index
    events coalesce). Values-first + polymorphic: raw arrays, Stream, or graph Node.

    No-index (positional) inputs are treated as aligned clocks (equal length
    required, lockstep); returns (aligned_values, None). Indexed inputs perform
    an as-of join aligned on each stream's index, returning (aligned_values, index).
    Stream inputs return a Stream. Node inputs return a Node.

    emit="when_all" (default) suppresses output until every input has a value;
    emit="on_any" emits from the first event (NaN for unseen inputs). If ``func``
    is given it is applied per row (``func(*row)``) after alignment.
    When a ``Stream`` is passed it carries its own index; ``index=`` applies only to raw arrays.
    """
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    if any(is_node(v) for v in values):
        if func is not None:
            raise ValueError(
                "combine_latest(func=...) is not supported in a DAG graph "
                "(graph ops are C++-only); apply a functor to the aligned output, "
                "e.g. Sub()(combine_latest(a, b))")
        return make_operator_node(combine_latest, values, {"emit": emit, "func": None})
    regime = _regime(values)
    streams = _to_streams(values, index)
    kind, idx, vals, positional = _streams_to_indexed(streams, "combine_latest")
    fn = _b._combine_latest_f64 if kind == "f64" else _b._combine_latest_i64
    out_index, aligned = fn(idx, vals, emit == "when_all")
    out_index, aligned = _collapse_last_per_index(out_index, aligned)
    result_index = None if positional else out_index
    if func is not None:
        aligned = np.array([func(*row) for row in aligned], dtype=np.float64)
    return _adapt(regime, aligned, result_index)


def combine_latest_iter(*values, index=None, emit="when_all"):
    """Yield coalesced (row, index) events: one per distinct index.

    Positional (no-index) inputs yield (row, None). Indexed inputs yield
    (row, index_value). Same coalescing semantics as combine_latest.
    """
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    streams = _to_streams(values, index)
    kind, idx, vals, positional = _streams_to_indexed(streams, "combine_latest")
    cls = _b._CombineLatestPuller_f64 if kind == "f64" else _b._CombineLatestPuller_i64
    puller = cls(idx, vals, emit == "when_all")
    cur_index, cur_row = None, None
    while True:
        event = puller.next()
        if event is None:
            break
        ev_index, ev_row = event   # puller.next() returns (index, row_tuple)
        if cur_index is not None and ev_index != cur_index:
            yield cur_row, (None if positional else cur_index)
        cur_index, cur_row = ev_index, ev_row
    if cur_index is not None:
        yield cur_row, (None if positional else cur_index)


async def pace(*values, index=None, speed=1.0, sleep=None):
    """Replay merged streams as an async event stream paced by index-deltas.

    Yields (value, index, source) in index order. Between consecutive events it
    awaits ``sleep(delta / speed)`` (where ``delta`` is the index delta) so
    wall-clock spacing tracks the index spacing. speed=inf disables pacing (backtest at max speed). Pacing never
    changes values or order. ``sleep`` is injectable for testing; defaults to
    asyncio.sleep. Requires a metric (subtractable) index.

    Positional (index=None): uses row-number as the pacing/ordering index; yields
    (value, None, source). Indexed: yields (value, index_value, source).
    """
    if speed <= 0:
        raise ValueError("pace: speed must be positive (or float('inf') for no pacing)")
    if sleep is None:
        sleep = asyncio.sleep
    infinite = speed == float("inf")
    kind, idx_list, vals_list, positional = _merge_to_indexed(values, index, "pace")
    cls = _b._MergePuller_f64 if kind == "f64" else _b._MergePuller_i64
    puller = cls(idx_list, vals_list)
    prev_index = None
    while True:
        event = puller.next()
        if event is None:
            return
        ev_index, ev_val, ev_source = event
        if not infinite and prev_index is not None:
            delta = ev_index - prev_index
            wait = delta / speed
            if wait > 0:
                await sleep(wait)
        prev_index = ev_index
        yield ev_val, (None if positional else ev_index), ev_source


def dropna(values, index=None, how="any"):
    """Drop events whose value is NaN. `values` may be 1-D (M,) or 2-D (M, N).

    how="any" (default) drops a row if any component is NaN; how="all" only if
    all are. Values-first + polymorphic: raw arrays, Stream, or graph Node.
    index=None means positional (no index allocation). Returns (values, index) /
    Stream / Node restricted to the surviving rows. Surviving values are returned
    as float64. Causal and cardinality-changing: batch and streaming drop the
    same rows.
    When ``values`` is a ``Stream`` it carries its own index; ``index=`` applies only to raw arrays.
    """
    if how not in ("any", "all"):
        raise ValueError('dropna: how must be "any" or "all"')
    if is_node(values):
        return make_operator_node(dropna, (values,), {"how": how})
    regime = "stream" if isinstance(values, Stream) else "raw"
    stream = values if isinstance(values, Stream) else Stream(values, index)
    vals = np.asarray(stream.values, dtype=np.float64)
    idx = stream.index
    nan = np.isnan(vals)
    if vals.ndim == 1:
        mask = ~nan
    else:
        mask = ~(nan.any(axis=1) if how == "any" else nan.all(axis=1))
    return _adapt(regime, vals[mask], None if idx is None else idx[mask])


def filter(values, predicate, index=None):
    """Keep events where predicate(row) is truthy.

    row is a scalar for a 1-D value stream, a 1-D array for a 2-D aligned stream.
    predicate is a Python callable (per-event), so heavy filtering should prefer
    dropna or numpy masks; filter is the general escape hatch. Values-first +
    polymorphic: raw arrays or Stream. Node first arg raises ValueError (no
    Python predicates in the graph engine).
    When ``values`` is a ``Stream`` it carries its own index; ``index=`` applies only to raw arrays.
    """
    if is_node(values):
        raise ValueError(
            "filter is not supported as a DAG graph node: the graph engine has "
            "no Python predicates (no lambda). Use dropna for NaN removal.")
    regime = "stream" if isinstance(values, Stream) else "raw"
    stream = values if isinstance(values, Stream) else Stream(values, index)
    vals = np.asarray(stream.values)
    idx = stream.index
    mask = np.fromiter((bool(predicate(row)) for row in vals),
                       dtype=bool, count=len(vals))
    return _adapt(regime, vals[mask], None if idx is None else idx[mask])


def dropna_iter(events, how="any"):
    """Streaming dropna over (value, index) tuples. value may be scalar or sequence.

    A positional live feed uses index=None. Surviving events are yielded as
    (value, index) with the original index passed through unchanged.
    """
    if how not in ("any", "all"):
        raise ValueError('dropna_iter: how must be "any" or "all"')
    for value, index in events:
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        nan = np.isnan(arr)
        drop = nan.any() if how == "any" else nan.all()
        if not drop:
            yield value, index


def filter_iter(events, predicate):
    """Streaming filter over (value, index) tuples.

    A positional live feed uses index=None. Surviving events are yielded as
    (value, index) with the original index passed through unchanged.
    """
    for value, index in events:
        if predicate(value):
            yield value, index


def split(values, sources, index=None, n=None):
    """Partition a merged tagged stream back into per-source (values, index) pairs.

    The inverse of merge: ``split(*merge(a_v, b_v, index=[a_k, b_k]))``
    reconstructs the inputs as ``[(a_v, a_k), (b_v, b_k)]``.
    Positional (index=None) returns ``(values_subset, None)`` per source.
    ``n`` sets how many output streams to produce (default: max(sources)+1); pass
    it explicitly to include sources that emitted nothing.
    """
    values = np.asarray(values)
    sources = np.asarray(sources)
    if n is None:
        n = int(sources.max()) + 1 if sources.size else 0
    elif sources.size and n <= int(sources.max()):
        raise ValueError(
            f"split: n={n} is too small for sources up to {int(sources.max())}; "
            "events would be dropped")
    if index is None:
        return [(values[sources == i], None) for i in range(n)]
    idx = np.asarray(index)
    return [(values[sources == i], idx[sources == i]) for i in range(n)]


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


def select(values, columns, index=None):
    """Pick column(s) from a wide (M, N) value stream.

    columns is an int (result is 1-D) or a sequence of ints (result is 2-D with
    those columns in order). Row count and index are unchanged (row-preserving
    shape op, not cardinality). Column indices must be in range and non-negative.
    Values-first + polymorphic: raw arrays, Stream, or graph Node.
    index=None means positional (no index allocation). Returns (values, index) /
    Stream / Node.
    When ``values`` is a ``Stream`` it carries its own index; ``index=`` applies only to raw arrays.
    """
    if is_node(values):
        return make_operator_node(select, (values,), {"columns": columns})
    regime = "stream" if isinstance(values, Stream) else "raw"
    stream = values if isinstance(values, Stream) else Stream(values, index)
    vals = np.asarray(stream.values, dtype=np.float64)
    idx = stream.index
    cols, scalar = _normalize_columns(columns)
    if vals.ndim == 1:
        width = 1
    else:
        width = vals.shape[1]
    for c in cols:
        if c >= width:
            raise ValueError(
                f"select: column {c} out of range for width {width}")
    if vals.ndim == 1:
        # width 1: only column 0 is valid; result mirrors input
        picked = vals if scalar else vals.reshape(-1, 1)
    else:
        picked = vals[:, cols[0]] if scalar else vals[:, cols]
    return _adapt(regime, picked, idx)   # index is unchanged (row-preserving)


def select_iter(events, columns):
    """Streaming select over (value, index) tuples. value is scalar or sequence.

    A positional live feed uses index=None. Projected events are yielded as
    (value, index) with the original index passed through unchanged.
    """
    cols, scalar = _normalize_columns(columns)
    for value, index in events:
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        for c in cols:
            if c >= arr.size:
                raise ValueError(
                    f"select_iter: column {c} out of range for width {arr.size}")
        if scalar:
            yield float(arr[cols[0]]), index
        else:
            yield [float(arr[c]) for c in cols], index


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


def _resample_validate(every, count, agg, label):
    if (every is None) == (count is None):
        raise ValueError("resample: pass exactly one of every= or count=")
    # Positivity guard: every=0 would reach the engine's floordiv(_, 0) -> a hard
    # SIGFPE crash; count<1 never completes a bucket. Reject both up front (this
    # runs before the Node dispatch, so it guards the graph path too).
    if every is not None and int(every) <= 0:
        raise ValueError("resample: every must be positive")
    if count is not None and int(count) < 1:
        raise ValueError("resample: count must be >= 1")
    if agg not in _RESAMPLE_AGGS:
        raise ValueError(f"resample: agg must be one of {_RESAMPLE_AGGS}")
    if label not in ("left", "right"):
        raise ValueError('resample: label must be "left" or "right"')


def resample(values, index=None, *, every=None, count=None, agg="last",
             origin=0, label="left"):
    """Causal windowed downsample of a 1-D value stream.

    Exactly one of `every` (fixed index-interval; buckets [origin+n*every,
    origin+(n+1)*every)) or `count` (fixed event-count). agg is one of
    first/last/min/max/sum/count/mean/ohlc (ohlc -> 4 columns). label "left"
    stamps the bucket start (by-index) or first index (by-count); "right" stamps the
    bucket end / last index. NaN values are ignored. Only non-empty buckets emit;
    the trailing partial bucket is emitted at end of input. Integer index-space.

    Values-first + polymorphic: raw arrays, Stream, or graph Node. The returned
    index is always the bar labels (a real array, never None) - even for a
    positional (no-index) input, which resamples by row position.

    Graph form: resample(node, ...) where node is a Node.
    When ``values`` is a ``Stream`` it carries its own index; ``index=`` applies only to raw arrays.
    """
    _resample_validate(every, count, agg, label)
    if is_node(values):
        return make_operator_node(resample, (values,), {
            "every": every, "count": count, "agg": agg,
            "origin": origin, "label": label})
    regime = "stream" if isinstance(values, Stream) else "raw"
    stream = values if isinstance(values, Stream) else Stream(values, index)
    vals = np.asarray(stream.values, dtype=np.float64)
    if vals.ndim != 1:
        raise ValueError("resample: expects a 1-D value stream")

    # Use explicit index or row positions when positional
    if stream.index is None:
        idx = np.arange(len(vals), dtype=np.int64)
    else:
        idx = np.asarray(stream.index)

    out_labels, out_vals = [], []
    acc = _ResampleAccum()

    def flush_bucket(label_index):
        out_labels.append(label_index)
        out_vals.append(acc.emit(agg))

    if every is not None:
        w = int(every)
        o = int(origin)
        started = False
        bucket = 0
        cur_label = 0
        for k, v in zip(idx.tolist(), vals.tolist()):
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
        first_index = 0
        last_index = 0
        for k, v in zip(idx.tolist(), vals.tolist()):
            k = int(k)
            if cib == 0:
                first_index = k
            last_index = k
            acc.add(v)
            cib += 1
            if cib == n:
                flush_bucket(first_index if label == "left" else last_index)
                acc.reset()
                cib = 0
        if cib > 0:
            flush_bucket(first_index if label == "left" else last_index)

    out_idx = np.array(out_labels, dtype=np.int64)
    if agg == "ohlc":
        out_v = (np.array(out_vals, dtype=np.float64).reshape(-1, 4)
                 if out_vals else np.empty((0, 4), dtype=np.float64))
    else:
        out_v = np.array(out_vals, dtype=np.float64)
    return _adapt(regime, out_v, out_idx)   # index is always real bar labels


def resample_iter(events, *, every=None, count=None, agg="last",
                  origin=0, label="left"):
    """Streaming resample over (value, index) tuples. Yields (value, label_index)."""
    _resample_validate(every, count, agg, label)
    acc = _ResampleAccum()
    if every is not None:
        w = int(every)
        o = int(origin)
        started = False
        bucket = 0
        cur_label = 0
        for v, k in events:
            k = int(k)
            nb = (k - o) // w
            if not started:
                started = True
                bucket = nb
                acc.reset()
                cur_label = (o + nb * w) if label == "left" else (o + (nb + 1) * w)
            elif nb != bucket:
                if acc.has:
                    yield acc.emit(agg), cur_label
                bucket = nb
                acc.reset()
                cur_label = (o + nb * w) if label == "left" else (o + (nb + 1) * w)
            acc.add(float(v))
        if acc.has:
            yield acc.emit(agg), cur_label
    else:
        n = int(count)
        cib = 0
        first_index = 0
        last_index = 0
        for v, k in events:
            k = int(k)
            if cib == 0:
                first_index = k
            last_index = k
            acc.add(float(v))
            cib += 1
            if cib == n:
                yield acc.emit(agg), (first_index if label == "left" else last_index)
                acc.reset()
                cib = 0
        if cib > 0:
            yield acc.emit(agg), (first_index if label == "left" else last_index)
