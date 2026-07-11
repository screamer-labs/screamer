"""Streaming operators (merge, combine_latest, replay, dropna, Filter, select,
split, resample).

Each operator dispatches on input type (Rule A): concrete data runs eager;
a lazy iterator of (value, index) events returns a lazy iterator. Builds and
runs C++ node graphs; dtype detection here chooses the int64 or float64
index-type instantiation, and the per-event work is all C++.
"""
import asyncio

import numpy as np

from . import screamer_bindings as _b
from .dag import is_node, make_operator_node

__all__ = [
    "Stream",
    "merge",
    "combine_latest",
    "replay",
    "dropna",
    "Filter",
    "select",
    "split",
    "resample",
]


class Stream:
    """A sequence of values with an optional ordering index and optional column names.

    ``values`` is a 1-D ``(T,)`` or 2-D ``(T, N)`` array. ``index`` is a 1-D array
    of length ``T``, or ``None``. ``None`` means positional (row number, arrival
    order) and stores nothing. The index is an ordering coordinate such as a
    timestamp or a tick counter, not a lookup key.

    ``columns`` is a tuple of string names for a 2-D ``values`` array, or ``None``
    for unlabelled (including all 1-D) streams. When set, ``stream["name"]`` and
    ``stream.column("name")`` return the corresponding 1-D column view.

    Backward-compatible unpacking: ``v, k = stream`` always unpacks to
    ``(stream.values, stream.index)``, so existing code that unpacked an eager
    ``resample(...)`` result as a 2-tuple continues to work unchanged.
    Integer subscript ``stream[0]`` returns ``values``; ``stream[1]`` returns
    ``index``.
    """
    __slots__ = ("values", "index", "columns")

    def __init__(self, values, index=None, columns=None):
        self.values = np.asarray(values)
        self.index = None if index is None else np.asarray(index)
        if self.index is not None and len(self.index) != len(self.values):
            raise ValueError("Stream: index and values must have the same length")
        if columns is not None:
            cols = tuple(columns)
            if self.values.ndim != 2:
                raise ValueError(
                    "Stream: columns requires a 2-D values array "
                    f"(got ndim={self.values.ndim})")
            if len(cols) != self.values.shape[1]:
                raise ValueError(
                    f"Stream: columns length {len(cols)} does not match "
                    f"values width {self.values.shape[1]}")
            self.columns = cols
        else:
            self.columns = None

    def __len__(self):
        """Return the number of rows (T). Note: ``len(stream) == T`` (row count)
        even though ``list(stream)`` yields 2 items ``[values, index]``."""
        return len(self.values)

    def __repr__(self):
        kind = "positional" if self.index is None else f"index={self.index!r}"
        cols = f", columns={self.columns!r}" if self.columns is not None else ""
        return f"Stream({self.values!r}, {kind}{cols})"

    def __iter__(self):
        """Yield ``(values, index)`` so that ``v, k = stream`` unpacks correctly.

        This is a deliberate 2-item iterator that preserves backward-compatible
        tuple-unpacking of eager ``resample(...)`` results.  ``list(stream)``
        returns ``[values_array, index_array]``.
        """
        return iter((self.values, self.index))

    def __getitem__(self, key):
        """Column or positional access.

        * ``stream["name"]`` - returns the 1-D column view by name (requires
          ``columns`` to be set; raises ``ValueError`` otherwise, ``KeyError``
          if the name is absent).
        * ``stream[0]`` - returns ``values`` (backward-compatible tuple index 0).
        * ``stream[1]`` - returns ``index``  (backward-compatible tuple index 1).
        """
        if isinstance(key, int):
            if key == 0:
                return self.values
            if key == 1:
                return self.index
            raise IndexError(f"Stream: integer index must be 0 or 1, got {key}")
        return self.column(key)

    def column(self, name):
        """Return the 1-D array for the named column.

        Raises ``ValueError`` if this stream has no column labels (``columns``
        is ``None``).  Raises ``KeyError`` if ``name`` is not in ``columns``.
        """
        if self.columns is None:
            raise ValueError(
                "Stream: cannot access column by name - this stream has no "
                "column labels (columns=None). Column labels are set for "
                "multi-column aggs such as ohlc.")
        try:
            col_idx = self.columns.index(name)
        except ValueError:
            raise KeyError(
                f"Stream: column {name!r} not found. "
                f"Available columns: {self.columns}")
        return self.values[:, col_idx]

    @classmethod
    def from_pandas(cls, obj):
        """Build a Stream from a pandas Series or DataFrame (data -> values,
        pandas index -> index). Note: a plain RangeIndex is kept as a numbered
        index, not converted to positional None."""
        return cls(obj.to_numpy(), np.asarray(obj.index))

    def to_pandas(self):
        """Return a pandas Series (1-D values) or DataFrame (2-D). A positional
        stream gets pandas' default RangeIndex. Column names are set when
        ``columns`` is not ``None``."""
        import pandas as pd
        if self.values.ndim == 1:
            return pd.Series(self.values, index=self.index)
        return pd.DataFrame(self.values, index=self.index,
                            columns=list(self.columns) if self.columns else None)


def _is_lazy_stream(x):
    """Rule A: a lazy stream is an iterator (has ``__next__``) that is NOT a
    concrete container (list, tuple, ndarray, or Stream). Generators and
    ``iter(...)`` qualify; concrete data does not."""
    return hasattr(x, "__next__") and not isinstance(x, (list, tuple, np.ndarray, Stream))


_EMPTY = object()   # sentinel: a lazy source produced no first item


def _classify_lazy_sources(values, who):
    """Classify N lazy sources as positional or indexed and return
    (positional, sources, kind).

    Peeks each source's first item: a bare scalar item means positional (its
    index is a per-source arrival counter), a ``(value, index)`` 2-tuple means
    indexed. Every source must be uniformly positional or indexed; a mix raises
    ValueError (matching the eager operators). ``sources`` are the original
    iterators re-chained with their peeked head, so no event is lost.

    ``kind`` is ``"i64"`` or ``"f64"`` and reflects the index dtype of the first
    non-empty indexed head (positional sources always use ``"i64"``).
    """
    import itertools
    iters = [iter(v) for v in values]
    heads, sources = [], []
    for it in iters:
        try:
            head = next(it)
        except StopIteration:
            head = _EMPTY
        heads.append(head)
        sources.append(it if head is _EMPTY else itertools.chain([head], it))
    kinds = {isinstance(h, tuple) and len(h) == 2 for h in heads if h is not _EMPTY}
    if len(kinds) > 1:
        raise ValueError(
            f"{who}: cannot mix positional (bare-value) and indexed "
            "((value, index)) lazy sources; give every source an index, or none")
    indexed = kinds.pop() if kinds else False   # all-empty -> positional (no rows)
    kind = "i64"
    if indexed:
        for h in heads:
            if h is not _EMPTY:
                _, idx_val = h
                kind = "f64" if isinstance(idx_val, (float, np.floating)) else "i64"
                break
    return (not indexed), sources, kind


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
    """Prepare merge/replay inputs for the C++ backend.

    Each input may be a raw value array or a ``Stream`` (which carries its own
    index; ``index=`` then applies only to the raw inputs). Positional (no index
    anywhere): uses per-stream row-number index (no equal-length check - unlike
    combine_latest). Indexed: uses the given index arrays. Mixing positional with
    indexed raises.

    Returns (kind, idx_list, vals_list, positional).
    """
    n = len(values)
    if n == 0:
        raise ValueError(f"{who}: needs at least one stream")
    streams = _to_streams(values, index)   # Stream passthrough; raw wrapped with index[i]
    has_idx = [s.index is not None for s in streams]
    if any(has_idx) and not all(has_idx):
        raise ValueError(
            f"{who}: cannot mix positional and indexed streams; give every "
            "stream an index, or none")
    vals = [np.ascontiguousarray(s.values, dtype=np.float64) for s in streams]
    if not any(has_idx):
        # Positional: use row-number per stream (no equal-length check for merge)
        idx_list = [np.arange(len(v), dtype=np.int64) for v in vals]
        return "i64", idx_list, vals, True
    kind, norm_index, _ = _normalize_streams([(s.index, s.values) for s in streams], who)
    return kind, norm_index, vals, False


def _merge_lazy_dispatch(values):
    """Deferred lazy dispatch: classify sources on first next(), then k-way merge.

    Classification (and peeking one head per source) is deferred to the first
    call to ``__next__``, so construction consumes nothing - matching the eager
    operators' lazy contract. Merge selection runs in the C++ MergeLazyPuller.
    Yields ``(value, index_or_None, source)`` byte-identical to the old Python loop.
    """
    positional, sources, kind = _classify_lazy_sources(values, "merge")
    cls = (_b._MergeLazyPuller_i64 if (positional or kind == "i64")
           else _b._MergeLazyPuller_f64)
    puller = cls(list(sources), positional)
    while True:
        ev = puller.next()
        if ev is None:
            return
        yield ev


def merge(*values, index=None):
    """Merge N value streams into one index-sorted (values, sources, index).

    Each stream must be individually sorted by its index. sources[i] is the
    source-stream index of event i. Ties break by stream order.

    Positional (index=None): uses row-number per stream; returned index is None.
    Unlike combine_latest, positional merge does NOT require equal lengths.
    Indexed (index=[idx0, idx1, ...]): merges by the given index arrays; returns
    the merged index array. Mixing positional and indexed raises ValueError.
    A Node input builds a graph node. Lazy iterator inputs return a lazy iterator
    of ``(value, index_or_None, source)`` events.
    """
    if any(is_node(v) for v in values):
        raise ValueError(
            "merge is not supported as a DAG graph node (it is input routing; "
            "feed streams to a Dag directly)")
    if values and all(_is_lazy_stream(v) for v in values):
        return _merge_lazy_dispatch(values)
    if any(_is_lazy_stream(v) for v in values):
        raise TypeError(
            "merge: cannot mix lazy iterator and concrete inputs; pass all "
            "generators or all arrays/Streams")
    kind, idx_list, vals_list, positional = _merge_to_indexed(values, index, "merge")
    fn = _b._merge_f64 if kind == "f64" else _b._merge_i64
    merged_index, merged_vals, sources = fn(idx_list, vals_list)
    return merged_vals, sources, (None if positional else merged_index)



def _combine_latest_zip_lazy(sources):
    """Positional (aligned-clock) combine_latest over lazy bare-value sources.

    Strict lockstep: every source must have equal length, matching the eager
    positional contract. Yields ``(row_tuple, None)``. Raises ValueError at the
    first length mismatch (lengths are unknowable until the streams run out).
    """
    import itertools
    for tup in itertools.zip_longest(*sources, fillvalue=_EMPTY):
        if any(x is _EMPTY for x in tup):
            raise ValueError(
                "combine_latest: positional (no-index) lazy sources must have "
                "equal length (aligned clocks); source lengths differ")
        yield tuple(float(x) for x in tup), None


def _combine_latest_asof_lazy(sources, emit):
    """Indexed combine_latest over lazy (value, index) sources: as-of alignment
    driven by the C++ combine_latest node through the Stage-2 lazy Dag. Yields
    ``(row_tuple, index)`` - identical to the batch combine_latest.

    The C++ node delays emission by one index step and collapses same-index
    events, so no external per-index deduplication is needed here. A non-integer
    index is rejected by the int64-indexed engine (_LazyDag._pull), which is
    correct: truncating a fractional index before alignment would collapse
    distinct indices and diverge from batch.
    """
    from .dag import Input, Dag
    ins = [Input(f"_cl{i}") for i in range(len(sources))]
    dag = Dag(inputs=ins, outputs=[combine_latest(*ins, emit=emit)])
    yield from dag(*sources)


def _combine_latest_lazy(values, emit):
    """Dispatch gate for lazy combine_latest: classify sources on first next().

    Classification (and peeking one head per source) is deferred to the first
    call to ``__next__``, so construction consumes nothing. Positional sources
    (bare scalars) route to strict-lockstep zip; indexed ((value, index))
    sources route to the as-of Dag path.
    """
    positional, sources, _ = _classify_lazy_sources(values, "combine_latest")
    if positional:
        yield from _combine_latest_zip_lazy(sources)
    else:
        yield from _combine_latest_asof_lazy(sources, emit)


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
    if values and all(_is_lazy_stream(v) for v in values):
        if func is not None:
            raise ValueError(
                "combine_latest(func=...) is not supported for lazy iterator "
                "inputs; apply the function to the aligned output instead")
        return _combine_latest_lazy(values, emit)
    if any(_is_lazy_stream(v) for v in values):
        raise TypeError(
            "combine_latest: cannot mix lazy iterator and concrete inputs; pass "
            "all generators or all arrays/Streams")
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


async def replay(*values, index=None, speed=1.0, sleep=None):
    """Replay merged streams as an async event stream paced by index-deltas.

    Each input may be a raw value array or a ``Stream``. Yields (value, index,
    source) in index order. Between consecutive events it awaits
    ``sleep(delta / speed)`` (where ``delta`` is the index delta) so wall-clock
    spacing tracks the index spacing. speed=inf disables pacing (backtest at max
    speed). Pacing never changes values or order. ``sleep`` is injectable for
    testing; defaults to asyncio.sleep. Requires a metric (subtractable) index.

    Positional (index=None): uses row-number as the pacing/ordering index; yields
    (value, None, source). Indexed: yields (value, index_value, source).
    """
    if any(is_node(v) for v in values):
        raise ValueError(
            "replay is not supported as a DAG graph node (it is input routing; "
            "feed streams to a Dag directly)")
    if speed <= 0:
        raise ValueError("replay: speed must be positive (or float('inf') for no pacing)")
    if sleep is None:
        sleep = asyncio.sleep
    infinite = speed == float("inf")
    kind, idx_list, vals_list, positional = _merge_to_indexed(values, index, "replay")
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
    if _is_lazy_stream(values):
        return _dropna_lazy(values, how)
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


class Filter:
    """2-input mask gate: keep each data value whose aligned mask is nonzero
    (zero or NaN drops). The mask is an ordinary stream built from upstream
    comparison or logic ops, e.g. ``Filter()(x, GreaterThan()(x, 0.0))``.
    No Python predicate or callback.

    Gate rule: a mask value of zero or NaN drops the aligned data value; any
    other mask value (positive, negative, non-NaN) keeps it. A NaN data value
    passes through unmodified when its aligned mask is nonzero.

    All three regimes (batch, lazy, graph) are driven by the same C++
    FilterNode and return byte-identical results.

    Batch (arrays or Streams)::

        survivors, idx = Filter()(data_array, mask_array)

    Lazy (iterators of (value, index) pairs)::

        for val, idx in Filter()(iter(data_events), iter(mask_events)):
            ...

    Graph (Node inputs)::

        d, m = Input("d"), Input("m")
        dag = Dag(inputs=[d, m], outputs=[Filter()(d, m)])
        survivors, idx = dag(data_array, mask_array)
    """

    __name__ = "Filter"

    def __call__(self, data, mask):
        if is_node(data) or is_node(mask):
            return make_operator_node(Filter, (data, mask), {})
        from .dag import Input, Dag
        d, m = Input("data"), Input("mask")
        dag = Dag(inputs=[d, m], outputs=[Filter()(d, m)])
        return dag(data, mask)


def _dropna_lazy(events, how="any"):
    """Streaming dropna over (value, index) tuples. value may be scalar or sequence.

    A positional live feed uses index=None. Surviving events are yielded as
    (value, index) with the original index passed through unchanged.
    """
    if how not in ("any", "all"):
        raise ValueError('dropna: how must be "any" or "all"')
    for value, index in events:
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        nan = np.isnan(arr)
        drop = nan.any() if how == "any" else nan.all()
        if not drop:
            yield value, index


def split(values, sources, index=None, n=None):
    """Partition a merged tagged stream back into per-source streams.

    The inverse of merge. ``values`` may be a raw value array or a ``Stream``
    (which carries its own index); ``sources`` is always passed separately.
    Raw in -> a list of ``(values, index)`` pairs; a ``Stream`` in -> a list of
    ``Stream`` objects, for type consistency:
    ``split(*merge(a_v, b_v, index=[a_k, b_k]))`` reconstructs ``[(a_v, a_k),
    (b_v, b_k)]``. Positional (index=None) uses ``None`` for the per-source index.
    ``n`` sets how many output streams to produce (default: max(sources)+1); pass
    it explicitly to include sources that emitted nothing.
    """
    as_stream = isinstance(values, Stream)
    if as_stream:
        index = values.index          # the Stream carries its own index
        values = values.values
    values = np.asarray(values)
    sources = np.asarray(sources)
    if n is None:
        n = int(sources.max()) + 1 if sources.size else 0
    elif sources.size and n <= int(sources.max()):
        raise ValueError(
            f"split: n={n} is too small for sources up to {int(sources.max())}; "
            "events would be dropped")
    if index is None:
        parts = [(values[sources == i], None) for i in range(n)]
    else:
        idx = np.asarray(index)
        parts = [(values[sources == i], idx[sources == i]) for i in range(n)]
    if as_stream:
        return [Stream(v, k) for v, k in parts]
    return parts


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
    if _is_lazy_stream(values):
        return _select_lazy(values, columns)
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


def _select_lazy(events, columns):
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
                    f"select: column {c} out of range for width {arr.size}")
        if scalar:
            yield float(arr[cols[0]]), index
        else:
            yield [float(arr[c]) for c in cols], index


_RESAMPLE_AGGS = ("first", "last", "min", "max", "sum", "count", "mean", "ohlc",
                  "ohlcv", "ohlcv2")
_RESAMPLE_FILLS = ("skip", "nan", "carry")
_OHLC_COLUMNS  = ("open", "high", "low", "close")
_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_OHLCV2_COLUMNS = ("open", "high", "low", "close", "buy_vol", "sell_vol")


def _resample_via_cpp(feed, *, every, count, agg, origin, label, fill="skip"):
    """Run resample on the C++ engine via a one-node Dag, for batch OR lazy input.

    Builds the minimal ``Input -> Resample`` graph the Node regime already uses,
    then defers to ``dag(feed)``. Rule A on the Dag decides the mode: a concrete
    ``(vals, idx)`` pair runs batch and returns ``(out_values, out_index)``; a lazy
    iterator of ``(value, index)`` events returns a lazy iterator of
    ``(bar_value, bar_label)``. No Python windowing - all bucketing and NaN-ignore
    accumulation happens in the C++ core.
    """
    from .dag import Input, Dag
    src = Input("x")
    node = resample(src, every=every, count=count, agg=agg,
                    origin=origin, label=label, fill=fill)
    dag = Dag([src], [node])
    return dag(feed)


def _resample_validate(every, count, agg, label, fill="skip"):
    if (every is None) == (count is None):
        raise ValueError("resample: pass exactly one of every= or count=")
    # Positivity guard: every=0 would reach the engine's floordiv(_, 0) -> a hard
    # SIGFPE crash; count<1 never completes a bucket. Reject both up front (this
    # runs before the Node dispatch, so it guards the graph path too).
    if every is not None and int(every) <= 0:
        raise ValueError("resample: every must be positive")
    if count is not None and int(count) < 1:
        raise ValueError("resample: count must be >= 1")
    # agg may be a builtin string, an arbitrary functor reducer (an EvalOp), or
    # a dict of {name: str|functor} entries. Only string scalars are validated
    # here; dict contents are validated in _resample_dict.
    if isinstance(agg, str) and agg not in _RESAMPLE_AGGS:
        raise ValueError(f"resample: agg must be one of {_RESAMPLE_AGGS}")
    if label not in ("left", "right"):
        raise ValueError('resample: label must be "left" or "right"')
    if fill not in _RESAMPLE_FILLS:
        raise ValueError(f"resample: fill must be one of {_RESAMPLE_FILLS}")


def _resample_dict(vals, idx, agg_dict, *, every, count, origin, label,
                   fill="skip"):
    """Run each sub-reducer in a dict agg independently over the same bucketing.

    Each entry ``{name: str|functor}`` is run through ``_resample_via_cpp``
    with the shared ``every``/``count``/``origin``/``label`` params.  Because all
    sub-resamples see the same input and the same bucketing they produce identical
    bar labels and the same number of bars, so their 1-D result columns align and
    can be horizontally stacked.

    Python only stacks the results and attaches names (marshalling); all numeric
    bucketing and accumulation runs in the C++ engine.

    Multi-column sub-aggs (``"ohlc"`` or a functor with ``num_outputs > 1``) are
    rejected with a ``ValueError`` in v1; each dict entry must produce exactly
    one column.  Use ``agg="ohlc"`` directly for a labelled 4-column ``Stream``.
    """
    if not agg_dict:
        raise ValueError("resample: agg dict must not be empty")

    out_cols = []
    out_idx = None

    for name, sub_agg in agg_dict.items():
        # Validate the sub-agg and reject multi-column entries.
        if isinstance(sub_agg, str):
            if sub_agg not in _RESAMPLE_AGGS:
                raise ValueError(
                    f"resample: dict agg[{name!r}]: unknown string agg {sub_agg!r}; "
                    f"must be one of {_RESAMPLE_AGGS}")
            if sub_agg == "ohlc":
                raise ValueError(
                    f"resample: dict agg[{name!r}]: 'ohlc' produces 4 columns and "
                    "cannot be used as a dict entry (v1: each entry must produce "
                    "exactly 1 column). Use agg='ohlc' directly for a labelled "
                    "4-column Stream.")
            if sub_agg in ("ohlcv", "ohlcv2"):
                raise ValueError(
                    f"resample: dict agg[{name!r}]: '{sub_agg}' requires a 2-column "
                    "input and cannot be used as a dict entry. "
                    f"Use agg='{sub_agg}' directly with a [price, volume] input.")
        else:
            # Functor: check num_outputs (exposed from EvalOp.n_out()).
            n_out = getattr(sub_agg, "num_outputs", 1)
            if n_out != 1:
                raise ValueError(
                    f"resample: dict agg[{name!r}]: functor produces {n_out} columns "
                    "but each dict entry must produce exactly 1 column (v1 restriction). "
                    "Use agg=functor directly for multi-column output.")

        # Delegate to the C++ engine - no Python numeric loop.
        out_v, sub_idx = _resample_via_cpp(
            (vals, idx), every=every, count=count, agg=sub_agg,
            origin=origin, label=label, fill=fill)
        out_cols.append(out_v)
        if out_idx is None:
            out_idx = sub_idx

    # All sub-aggs produce the same number of bars (same bucketing + same input),
    # so column_stack is safe: shape (N,) per column -> (N, K) matrix.
    stacked = np.column_stack(out_cols)
    return Stream(stacked, out_idx, columns=tuple(agg_dict.keys()))


def _resample_ohlcv(vals_2d, idx, agg, *, every, count, origin, label,
                    fill="skip"):
    """Orchestrate ohlcv or ohlcv2 by composing existing C++ reducers.

    Splits the 2-column input [price, volume|signed_volume] into two 1-D arrays
    and runs independent sub-resamples that all share the same bucketing
    parameters. Because the input, every/count, origin, and label are identical
    across sub-calls, bar labels and bar counts are guaranteed to be equal -
    making ``column_stack`` on the results safe.

    All numeric compute runs in C++ (OHLC via the existing ``ohlc`` reducer,
    volume via ``sum``, buy/sell via ``PosPart``/``NegPart`` then ``sum``).
    Python only splits columns, orchestrates the sub-calls, and attaches labels.
    """
    from screamer import PosPart, NegPart

    price  = np.ascontiguousarray(vals_2d[:, 0], dtype=np.float64)
    volume = np.ascontiguousarray(vals_2d[:, 1], dtype=np.float64)

    # OHLC for price (C++ ohlc reducer -> 4 columns)
    ohlc_vals, out_idx = _resample_via_cpp(
        (price, idx), every=every, count=count, agg="ohlc",
        origin=origin, label=label, fill=fill)

    if agg == "ohlcv":
        # Total signed-volume sum (C++ sum reducer -> 1 column)
        vol_vals, _ = _resample_via_cpp(
            (volume, idx), every=every, count=count, agg="sum",
            origin=origin, label=label, fill=fill)
        stacked = np.column_stack([ohlc_vals, vol_vals])
        return Stream(stacked, out_idx, columns=_OHLCV_COLUMNS)

    # ohlcv2: buy_vol = sum(PosPart(signed_vol)), sell_vol = sum(NegPart(signed_vol))
    # PosPart/NegPart are C++ functors; applying them to the array runs in C++.
    buy_arr  = np.asarray(PosPart()(volume), dtype=np.float64)
    sell_arr = np.asarray(NegPart()(volume), dtype=np.float64)

    buy_vals, _  = _resample_via_cpp(
        (buy_arr,  idx), every=every, count=count, agg="sum",
        origin=origin, label=label, fill=fill)
    sell_vals, _ = _resample_via_cpp(
        (sell_arr, idx), every=every, count=count, agg="sum",
        origin=origin, label=label, fill=fill)

    stacked = np.column_stack([ohlc_vals, buy_vals, sell_vals])
    return Stream(stacked, out_idx, columns=_OHLCV2_COLUMNS)


def resample(values, index=None, *, every=None, count=None, agg="last",
             origin=0, label="left", fill="skip"):
    """Causal windowed downsample of a 1-D value stream.

    Exactly one of ``every`` or ``count`` bounds the bars:

    * ``every=W`` buckets along the **index**: bar ``n`` is the half-open interval
      ``[origin+n*W, origin+(n+1)*W)`` (boundaries anchored at ``origin``, default 0,
      i.e. multiples of ``W`` - not at the first tick). Equal width on the index,
      variable ticks per bar; a tick exactly on a boundary starts the later bar.
    * ``count=N`` buckets by **arrival order**: a bar closes every ``N`` events and
      does not consult the index values for boundaries. Equal ticks per bar,
      variable index width; a bar may straddle an arbitrary index gap.

    ``index`` is optional in both modes; omit it to bucket/label by row position
    (``0, 1, 2, ...``). ``count`` does not need it for boundaries; ``every`` uses it
    as the timeline.

    ``agg`` controls the per-bucket aggregation:

    * **string**: one of ``first``, ``last``, ``min``, ``max``, ``sum``,
      ``count``, ``mean``, ``ohlc``.  ``ohlc`` returns 4 columns
      ``(open, high, low, close)`` labelled on the returned ``Stream``.
    * **functor**: any :class:`screamer.EvalOp` reducer (e.g.
      ``ExpandingSkew()``).  The functor is ``reset()`` at each bar boundary
      and fed every in-bar sample; its last output before the close is emitted.
      A single-output functor returns a 1-D ``Stream``; a multi-output functor
      returns an unlabelled 2-D ``Stream``.  The functor must accept exactly
      1 input (single-value stream); a multi-input functor raises at runtime.
    * **dict** ``{name: agg, ...}``: runs several reducers over the same
      bucketing and returns a labelled ``Stream`` whose ``.columns`` are the dict
      keys (insertion order).  Each entry must produce exactly 1 column; ``"ohlc"``
      and multi-output functors are rejected (use ``agg="ohlc"`` directly for
      4-column output).  Two forms:

      - **eager** (raw arrays / ``Stream``): each value is a string or functor
        reducer applied to the single value stream, e.g.
        ``{"open": "first", "vol": "sum"}``.
      - **graph / lazy** (``resample(t, agg={...})`` where the values are Nodes):
        each value is a lazy expression ``Reducer()(sub_expr)`` whose top node is
        the per-bar reducer and whose single input is the upstream port (per-tick
        transforms live there), e.g.
        ``{"buy": ExpandingSum()(PosPart()(vol))}``.  The first positional argument
        ``t`` is the clock; all columns share one bar clock, so they cannot drift.
        Place the result in a ``Dag`` and bind data at call time.

    ``label`` picks each bar's index. For ``every=`` it is the **grid edge**
    (``origin+n*W`` for ``"left"``, ``origin+(n+1)*W`` for ``"right"``), the
    interval boundary, which need not be an actual tick. For ``count=`` it is an
    **actual tick index** (the bar's first tick for ``"left"``, its last for
    ``"right"``).  NaN values are ignored.

    ``fill`` controls empty **internal** buckets (a gap between two events where
    one or more buckets have no samples). It is meaningful only under ``every=``;
    with ``count=`` a bar is defined by having ``N`` events, so empty bars cannot
    occur and ``fill`` has no effect.

    * ``"skip"`` (default): no row for an empty bucket (legacy behavior).
    * ``"nan"``: an all-NaN row at each empty bucket's label.
    * ``"carry"``: repeat the previous emitted row's values verbatim (v1: the
      whole row is carried, including sum/volume-like columns; a per-column
      "carry price / zero volume" refinement is out of scope).

    Only internal gaps are filled. Leading buckets before the first event and
    trailing buckets after the last are not synthesized here (output starts at the
    first tick's bucket; trailing empties need the clock/``advance()`` mechanism).
    ``"carry"`` repeats the whole previous row verbatim across all columns (so an
    empty bar carries a count/sum column's prior value, not 0; use ``"nan"`` if
    that is wrong for a column), and skips a bucket that has no prior emitted row.
    A filled bucket emits a synthetic row without feeding or resetting the reducer,
    so a functor reducer starts each real bar clean. The trailing partial bucket is
    still emitted once at end of input. Integer index-space.

    Values-first + polymorphic: raw arrays, Stream, or graph Node. The returned
    index is always the bar labels (a real array, never None) - even for a
    positional (no-index) input, which resamples by row position.

    Graph form: resample(node, ...) where node is a Node.
    When ``values`` is a ``Stream`` it carries its own index; ``index=`` applies only to raw arrays.
    """
    _resample_validate(every, count, agg, label, fill)
    if is_node(values):
        if isinstance(agg, dict):
            # Lazy multi-column bars: each value is Reducer()(sub_expr). `values` is
            # the clock/timeline `t`. Split each expr into (reducer, port); one shared
            # clock drives empty time-bar finalization in every= mode.
            if not agg:
                raise ValueError("resample: agg dict is empty")
            names, ports, reducers = [], [], []
            for nm, expr in agg.items():
                red, port = _split_reducer_expr(nm, expr)
                names.append(nm)
                ports.append(port)
                reducers.append(red)
            return multi_resample(ports, reducers, clock=values, every=every,
                                  count=count, origin=origin, label=label,
                                  fill=fill, columns=tuple(names))
        if agg in ("ohlcv", "ohlcv2"):
            raise ValueError(
                f"resample: the agg='{agg}' string shorthand is eager-only and is "
                "not supported in the graph (Node) regime. In a graph, build the "
                "columns with a lazy agg dict instead, e.g. agg={'open': "
                "First()(price), 'buy': ExpandingSum()(PosPart()(vol)), ...}.")
        return make_operator_node(resample, (values,), {
            "every": every, "count": count, "agg": agg,
            "origin": origin, "label": label, "fill": fill})
    if _is_lazy_stream(values):
        # Rule A: a lazy iterator of (value, index) events -> a lazy iterator of
        # (bar_value, bar_label). Drive the same C++ resample node as batch through
        # the Stage-2 lazy Dag; no Python windowing accumulator runs here.
        if isinstance(agg, dict) or agg in ("ohlcv", "ohlcv2"):
            raise ValueError(
                "resample(<iterator>) supports string and functor scalar aggs "
                "only; dict and ohlcv/ohlcv2 aggs are eager-only. Materialize the "
                "stream to an array for those, or build the columns inside a Dag.")
        return _resample_via_cpp(values, every=every, count=count, agg=agg,
                                 origin=origin, label=label, fill=fill)
    stream = values if isinstance(values, Stream) else Stream(values, index)
    vals = np.asarray(stream.values, dtype=np.float64)
    # ohlcv / ohlcv2 require exactly 2 input columns [price, volume|signed_volume].
    if agg in ("ohlcv", "ohlcv2"):
        if vals.ndim != 2 or vals.shape[1] != 2:
            raise ValueError(
                f"resample: agg='{agg}' requires exactly 2 columns "
                f"[price, {'volume' if agg == 'ohlcv' else 'signed_volume'}]; "
                f"got shape {vals.shape}. Pass np.column_stack([price, volume]).")
    elif vals.ndim != 1:
        raise ValueError("resample: expects a 1-D value stream")

    # Use explicit index or row positions when positional
    if stream.index is None:
        idx = np.arange(len(vals), dtype=np.int64)
    else:
        idx = np.asarray(stream.index)

    # Dict agg: run each sub-reducer over the shared bucketing and stack columns.
    # Python only stacks the results and attaches names; all numeric reduction
    # runs in the C++ engine via the existing single-agg path.
    if isinstance(agg, dict):
        return _resample_dict(vals, idx, agg, every=every, count=count,
                              origin=origin, label=label, fill=fill)

    # ohlcv / ohlcv2: orchestrate existing C++ reducers over a 2-column input.
    # Python splits the columns, runs sub-resamples, and stacks; all numeric
    # bucketing and accumulation runs in C++ as with the single-column path.
    if agg in ("ohlcv", "ohlcv2"):
        return _resample_ohlcv(vals, idx, agg, every=every, count=count,
                               origin=origin, label=label, fill=fill)

    # Delegate all bucketing/accumulation to the C++ engine (C++-first: no
    # Python numeric loop). Builds a one-node graph and runs it in batch.
    out_v, out_idx = _resample_via_cpp(
        (vals, idx), every=every, count=count, agg=agg, origin=origin, label=label,
        fill=fill)
    # Attach column names for multi-column aggs; labels are pure Python marshalling.
    cols = _OHLC_COLUMNS if agg == "ohlc" else None
    # Both raw and Stream regimes return a Stream; Stream is unpackable as
    # (values, index) for backward-compatible tuple unpacking.
    return Stream(out_v, out_idx, columns=cols)


def _split_reducer_expr(name, expr):
    """A dict-agg value must be ``Reducer()(sub_expr)``: a functor Node with exactly one
    input. Returns ``(reducer_evalop, port_node)``. Raises a clear error otherwise."""
    if not is_node(expr):
        raise ValueError(
            f"resample: agg[{name!r}] must be a lazy expression like "
            f"First()(price); got {type(expr).__name__}. In the graph regime the "
            f"dict values are code fragments (functors applied to Nodes), not strings.")
    op = expr.op
    # A functor node's op is the EvalOp instance (not a tuple).
    # An operator/input node's op is a tuple: ("input", name) or ("operator", fn, kwargs).
    if isinstance(op, tuple):
        kind = op[0] if op else "?"
        raise ValueError(
            f"resample: agg[{name!r}] must be a single reducer functor applied to a "
            f"stream (e.g. ExpandingSum()(PosPart()(vol))); its top node is a "
            f"{kind!r} node, not a reducer functor.")
    if len(expr.inputs) != 1:
        raise ValueError(
            f"resample: agg[{name!r}] reducer must take exactly one input stream; "
            f"got {len(expr.inputs)}.")
    return op, expr.inputs[0]


def multi_resample(inputs, reducers, clock=None, every=None, count=None, origin=0,
                   label="left", fill="skip", columns=None):
    """Low-level multi-column bar node: N port streams, N per-bar reducers, one clock.

    ``inputs[i]`` (a Node) is reduced by ``reducers[i]`` (an EvalOp) within each
    bar; the node emits one aligned row per bar with the reducers' outputs
    concatenated. Transforms belong upstream in ``inputs[i]``. Column labels are
    attached by the caller (see ``bars``). This is the graph (Node) primitive;
    place it in a Dag and bind data at call time.

    ``count=N`` buckets by N DISTINCT ticks (indices): several ports pushing at the
    same index count as ONE tick, and a full bar closes on the next new tick.

    ``clock`` (optional Node): a timestamp/clock stream that drives bucketing. A
    clock tick crossing a bucket boundary closes the current bar even with no
    trades, so empty time-bars finalize straight from the data (ByIndex/``every=``
    only). Omit it to bucket purely by the columns' own shared ticks.

    ``columns`` (optional tuple of str): column labels for the output. Not passed to
    the C++ engine; used by the Dag to label the returned Stream.
    """
    if len(inputs) != len(reducers):
        raise ValueError(
            "multi_resample: inputs and reducers must have equal length")
    node_inputs = list(inputs)
    if clock is not None:
        if not is_node(clock):
            raise ValueError("multi_resample: clock must be a graph Node")
        node_inputs.append(clock)   # clock is the LAST input (inferred as the clock port)
    if not all(is_node(x) for x in node_inputs):
        raise ValueError("multi_resample: every input must be a graph Node")
    return make_operator_node(multi_resample, tuple(node_inputs), {
        "reducers": list(reducers), "every": every, "count": count,
        "origin": origin, "label": label, "fill": fill, "columns": columns})


