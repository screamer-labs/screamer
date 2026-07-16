"""Streaming operators (merge, combine_latest, dropna, Filter, select,
split, resample).

Each operator dispatches on input type (Rule A): concrete data runs eager;
a lazy iterator of (value, index) events returns a lazy iterator. Builds and
runs C++ node graphs; dtype detection here chooses the int64 or float64
index-type instantiation, and the per-event work is all C++.
"""
import datetime
import re

import numpy as np

from . import screamer_bindings as _b
from .dag import is_node, make_operator_node, _is_vi_pair

__all__ = [
    "to_pandas",
    "from_pandas",
    "Filter",
    "split",
    "Merge",
    "CombineLatest",
    "Dropna",
    "Select",
    "Resample",
]


def _is_lazy_stream(x):
    """Rule A: a lazy stream is an iterator (has ``__next__``) that is NOT a
    concrete container (list, tuple, or ndarray). Generators and
    ``iter(...)`` qualify; concrete data does not."""
    return hasattr(x, "__next__") and not isinstance(x, (list, tuple, np.ndarray))


def _as_vi(x, idx):
    """Normalize a data input to (values_ndarray, index_or_None).

    Accepts:
    - a bare array (positional or with explicit idx) -> (np.asarray(x), idx)
    - a (values, index) 2-tuple -> pass through
    """
    if _is_vi_pair(x):
        return x[0], x[1]
    return np.asarray(x), idx


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


def _to_streams(inputs, index):
    """Normalize each input to a (values, index) tuple. `index` is None (all
    positional) or a list aligned with inputs (per-stream index array or None)."""
    if index is not None and len(index) != len(inputs):
        raise ValueError("index list length must match the number of streams")
    return [_as_vi(x, None if index is None else index[i])
            for i, x in enumerate(inputs)]


def to_pandas(values, index=None, columns=None):
    """Return a pandas Series (1-D values) or DataFrame (2-D).

    A positional stream (index=None) gets pandas' default RangeIndex.
    Column names are set when columns is not None.
    """
    import pandas as pd
    values = np.asarray(values)
    if values.ndim == 1:
        return pd.Series(values, index=index)
    return pd.DataFrame(values, index=index,
                        columns=list(columns) if columns else None)


def from_pandas(obj):
    """Build a (values, index) tuple from a pandas Series or DataFrame.

    data -> values (numpy array), pandas index -> index array. Note: pandas
    always materializes an index (a RangeIndex for a default one), so a
    positional stream does not round-trip to index=None through
    to_pandas/from_pandas - it comes back with an explicit 0,1,2,... index.
    """
    return np.asarray(obj.to_numpy()), np.asarray(obj.index)


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


def _streams_to_indexed(streams, who):
    """(kind, index_list, vals_list, positional). Uniform positional or indexed;
    no-index requires equal length; mixing positional and indexed raises."""
    indexed = [s[1] is not None for s in streams]
    if any(indexed) and not all(indexed):
        raise ValueError(
            f"{who}: cannot align positional and indexed streams; give every "
            "stream an index, or none")
    vals = [np.ascontiguousarray(s[0], dtype=np.float64) for s in streams]
    if not any(indexed):
        lens = {len(s[0]) for s in streams}
        if len(lens) != 1:
            raise ValueError(
                f"{who}: streams have no index, so they are assumed aligned - "
                "lengths must match, or provide an index to align different clocks")
        idx = [np.arange(len(streams[0][0]), dtype=np.int64) for _ in streams]
        return "i64", idx, vals, True
    kind, idx, _ = _normalize_streams([(s[1], s[0]) for s in streams], who)
    return kind, idx, vals, False


def _merge_to_indexed(values, index, who):
    """Prepare merge inputs for the C++ backend.

    Each input may be a raw value array or a ``(values, index)`` tuple (which
    carries its own index; ``index=`` then applies only to the raw inputs).
    Positional (no index anywhere): uses per-stream row-number index (no
    equal-length check - unlike combine_latest). Indexed: uses the given index
    arrays. Mixing positional with indexed raises.

    Returns (kind, idx_list, vals_list, positional).
    """
    n = len(values)
    if n == 0:
        raise ValueError(f"{who}: needs at least one stream")
    streams = _to_streams(values, index)   # (values, index) passthrough; raw wrapped with index[i]
    has_idx = [s[1] is not None for s in streams]
    if any(has_idx) and not all(has_idx):
        raise ValueError(
            f"{who}: cannot mix positional and indexed streams; give every "
            "stream an index, or none")
    vals = [np.ascontiguousarray(s[0], dtype=np.float64) for s in streams]
    if not any(has_idx):
        # Positional: use row-number per stream (no equal-length check for merge)
        idx_list = [np.arange(len(v), dtype=np.int64) for v in vals]
        return "i64", idx_list, vals, True
    kind, norm_index, _ = _normalize_streams([(s[1], s[0]) for s in streams], who)
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
            "merge is not supported as a Pipeline graph node (it is input routing; "
            "feed streams to a Pipeline directly)")
    if values and all(_is_lazy_stream(v) for v in values):
        return _merge_lazy_dispatch(values)
    if any(_is_lazy_stream(v) for v in values):
        raise TypeError(
            "merge: cannot mix lazy iterator and concrete inputs; pass all "
            "generators or all arrays")
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
    driven by the C++ combine_latest node through the Stage-2 lazy Pipeline. Yields
    ``(row_tuple, index)`` - identical to the batch combine_latest.

    The C++ node delays emission by one index step and collapses same-index
    events, so no external per-index deduplication is needed here. A non-integer
    index is rejected by the int64-indexed engine (_LazyDag._pull), which is
    correct: truncating a fractional index before alignment would collapse
    distinct indices and diverge from batch.
    """
    from .dag import Input, Pipeline
    ins = [Input(f"_cl{i}") for i in range(len(sources))]
    dag = Pipeline(inputs=ins, outputs=[combine_latest(*ins, emit=emit)])
    yield from dag(*sources)


def _combine_latest_lazy(values, emit):
    """Dispatch gate for lazy combine_latest: classify sources on first next().

    Classification (and peeking one head per source) is deferred to the first
    call to ``__next__``, so construction consumes nothing. Positional sources
    (bare scalars) route to strict-lockstep zip; indexed ((value, index))
    sources route to the as-of Pipeline path.
    """
    positional, sources, _ = _classify_lazy_sources(values, "combine_latest")
    if positional:
        yield from _combine_latest_zip_lazy(sources)
    else:
        yield from _combine_latest_asof_lazy(sources, emit)


def combine_latest(*values, index=None, emit="when_all"):
    """As-of latest-value join of N streams: one row per distinct index (same-index
    events coalesce). Values-first + polymorphic: raw arrays, (values, index)
    tuples, or graph Nodes.

    No-index (positional) inputs are treated as aligned clocks (equal length
    required, lockstep); returns (aligned_values, None). Indexed inputs perform
    an as-of join aligned on each stream's index, returning (aligned_values, index).
    Node inputs return a Node.

    emit="when_all" (default) suppresses output until every input has a value;
    emit="on_any" emits from the first event (NaN for unseen inputs). To transform
    each aligned row, apply a functor to the output, e.g. Sub()(combine_latest(a, b)).
    When a (values, index) tuple is passed it carries its own index; index=
    applies only to raw arrays.
    """
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    if any(is_node(v) for v in values):
        return make_operator_node(CombineLatest, values, {"emit": emit})
    if values and all(_is_lazy_stream(v) for v in values):
        return _combine_latest_lazy(values, emit)
    if any(_is_lazy_stream(v) for v in values):
        raise TypeError(
            "combine_latest: cannot mix lazy iterator and concrete inputs; pass "
            "all generators or all arrays")
    streams = _to_streams(values, index)
    kind, idx, vals, positional = _streams_to_indexed(streams, "combine_latest")
    fn = _b._combine_latest_f64 if kind == "f64" else _b._combine_latest_i64
    out_index, aligned = fn(idx, vals, emit == "when_all")   # coalesced in C++
    return aligned, (None if positional else out_index)



def _dropna_via_cpp(vals, idx, how):
    """Route eager dropna through the C++ DropNaNode via a Pipeline.

    vals: float64 ndarray, 1-D or 2-D
    idx: index array or None (positional)
    how: "any" | "all"

    Returns (out_vals, out_idx) where out_idx mirrors the caller's positional
    convention: None when the input index was None, otherwise the surviving
    subset of the input index.

    1-D: builds a one-input Pipeline([x], [dropna(x, how=how)]).
    2-D: packs N columns via combine_latest then drops via
    Pipeline([c0..cN], [dropna(combine_latest(c0..cN), how=how)]).
    All NaN-gate logic runs in the C++ DropNaNode; no numpy mask compute here.
    """
    from .dag import Input, Pipeline
    positional = idx is None
    real_idx = np.arange(len(vals), dtype=np.int64) if positional else idx

    if vals.ndim == 1:
        src = Input("x")
        dag = Pipeline([src], [dropna(src, how=how)])
        out_vals, out_idx = dag((vals, real_idx))
    else:
        N = vals.shape[1]
        col_inputs = [Input(f"c{i}") for i in range(N)]
        dag = Pipeline(col_inputs, [dropna(combine_latest(*col_inputs), how=how)])
        out_vals, out_idx = dag(*((vals[:, i], real_idx) for i in range(N)))

    return out_vals, (None if positional else out_idx)


def _dropna_lazy_cpp(events, how):
    """Route lazy dropna through the C++ DropNaNode via the Pipeline lazy path.

    Generator function: no events are consumed until the first next() call.
    Peeks the first event to determine dimensionality and index type.

    1-D events (scalar values): drives a one-input Pipeline.
    2-D events (multi-value rows): tees the stream into N per-column iterators
    and drives an N-input combine_latest + dropna Pipeline.  The N tee'd iterators
    are advanced in lockstep by _LazyDag, so the tee buffer is at most O(N)
    elements (constant in stream length - O(1) per event).

    Positional feeds (index=None) are converted to row-number indices
    internally and the None index is restored on output.
    """
    import itertools
    from .dag import Input, Pipeline

    try:
        head = next(events)
    except StopIteration:
        return

    head_val, head_idx = head
    all_events = itertools.chain([head], events)
    N = np.atleast_1d(np.asarray(head_val, dtype=np.float64)).size
    positional = head_idx is None

    if positional:
        def _add_rownum(evs):
            for i, (v, _k) in enumerate(evs):
                yield v, i
        working_events = _add_rownum(all_events)
    else:
        working_events = all_events

    if N == 1:
        # 1-D: ensure the value passed to _LazyDag._pull is a Python float
        def _as_scalar(evs):
            for v, k in evs:
                yield float(np.atleast_1d(np.asarray(v, dtype=np.float64))[0]), k
        src = Input("x")
        dag = Pipeline([src], [dropna(src, how=how)])
        inner = dag(_as_scalar(working_events))
    else:
        # 2-D: tee N copies; each copy is consumed once per event (lockstep)
        tees = itertools.tee(working_events, N)

        def _col(teed, col):
            for row_val, k in teed:
                r = np.atleast_1d(np.asarray(row_val, dtype=np.float64))
                yield float(r[col]), k

        col_iters = [_col(tees[i], i) for i in range(N)]
        col_inputs = [Input(f"c{i}") for i in range(N)]
        dag = Pipeline(col_inputs, [dropna(combine_latest(*col_inputs), how=how)])
        inner = dag(*col_iters)

    if positional:
        for v, _k in inner:
            yield v, None
    else:
        yield from inner


def dropna(values, index=None, how="any"):
    """Drop events whose value is NaN. `values` may be 1-D (M,) or 2-D (M, N).

    how="any" (default) drops a row if any component is NaN; how="all" only if
    all are. Values-first + polymorphic: raw arrays, (values, index) tuple, or
    graph Node.
    index=None means positional (no index allocation). Returns (values, index) /
    Node restricted to the surviving rows. Surviving values are returned
    as float64. Causal and cardinality-changing: batch and streaming drop the
    same rows.
    When ``values`` is a ``(values, index)`` tuple it carries its own index;
    ``index=`` applies only to raw arrays.
    """
    if how not in ("any", "all"):
        raise ValueError('dropna: how must be "any" or "all"')
    if is_node(values):
        return make_operator_node(Dropna, (values,), {"how": how})
    if _is_lazy_stream(values):
        return _dropna_lazy_cpp(values, how)
    vals, idx = _as_vi(values, index)
    vals = np.asarray(vals, dtype=np.float64)
    out_vals, out_idx = _dropna_via_cpp(vals, idx, how)
    return out_vals, out_idx


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

    Batch (arrays or (values, index) tuples)::

        survivors, idx = Filter()(data_array, mask_array)

    Lazy (iterators of (value, index) pairs)::

        for val, idx in Filter()(iter(data_events), iter(mask_events)):
            ...

    Graph (Node inputs)::

        d, m = Input("d"), Input("m")
        dag = Pipeline(inputs=[d, m], outputs=[Filter()(d, m)])
        survivors, idx = dag(data_array, mask_array)
    """

    __name__ = "Filter"

    def __call__(self, data, mask):
        if is_node(data) or is_node(mask):
            return make_operator_node(Filter, (data, mask), {})
        from .dag import Input, Pipeline
        d, m = Input("data"), Input("mask")
        dag = Pipeline(inputs=[d, m], outputs=[Filter()(d, m)])
        return dag(data, mask)


def split(values, sources, index=None, n=None):
    """Partition a merged tagged stream back into per-source streams.

    The inverse of merge. ``values`` may be a raw value array or a
    ``(values, index)`` tuple (which carries its own index); ``sources`` is
    always passed separately.
    Returns a list of ``(values, index)`` pairs. Positional (index=None) uses
    ``None`` for the per-source index.
    ``n`` sets how many output streams to produce (default: max(sources)+1); pass
    it explicitly to include sources that emitted nothing.
    """
    if _is_vi_pair(values):
        index = values[1]          # the tuple carries its own index
        values = values[0]
    values = np.ascontiguousarray(values, dtype=np.float64)
    sources = np.ascontiguousarray(sources, dtype=np.uint32)
    if n is None:
        n = int(sources.max()) + 1 if sources.size else 0
    elif sources.size and n <= int(sources.max()):
        raise ValueError(
            f"split: n={n} is too small for sources up to {int(sources.max())}; "
            "events would be dropped")
    if index is None:
        # positional: partition values only; a row-number index rides along the
        # C++ partition and is dropped to None on the way out.
        placeholder = np.arange(values.shape[0], dtype=np.int64)
        parts = _b._split_i64(values, sources, placeholder, n)
        return [(v, None) for (v, _k) in parts]
    kind, idx = _index_dtype_kind(index)
    fn = _b._split_f64 if kind == "f64" else _b._split_i64
    return list(fn(values, sources, idx, n))


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


def _select_via_cpp(vals, idx, cols, scalar):
    """Route eager select through the C++ SelectNode via a Pipeline.

    vals: float64 ndarray, 1-D or 2-D
    idx: index array or None (positional)
    cols: validated list of non-negative int column indices
    scalar: True when a bare int was given (result is 1-D)

    Returns (picked_vals, out_idx) where out_idx mirrors the input: None for
    positional, original idx array otherwise.

    1-D input: trivial passthrough or reshape; no SelectNode needed.
    2-D input: packs N columns via combine_latest then picks via
    Pipeline([c0..cN], [select(combine_latest(c0..cN), columns=columns)]).
    All column-pick logic runs in the C++ SelectNode; no numpy column-pick here.
    """
    from .dag import Input, Pipeline

    if vals.ndim == 1:
        # Width-1 passthrough: no column-pick compute, only scalar/list reshape.
        picked = vals if scalar else vals.reshape(-1, 1)
        return picked, idx

    # 2-D: route through SelectNode via Pipeline.
    positional = idx is None
    real_idx = np.arange(len(vals), dtype=np.int64) if positional else idx
    N = vals.shape[1]
    col_inputs = [Input(f"c{i}") for i in range(N)]
    columns_arg = cols[0] if scalar else cols
    dag = Pipeline(col_inputs, [select(combine_latest(*col_inputs), columns=columns_arg)])
    out_vals, out_idx = dag(*((vals[:, i], real_idx) for i in range(N)))
    return out_vals, (None if positional else out_idx)


def _select_lazy_cpp(events, columns):
    """Route lazy select through the C++ SelectNode via the Pipeline lazy path.

    Generator function: no events are consumed until the first next() call.
    Peeks the first event to determine dimensionality and index type.

    1-D events (scalar values): passthrough with column 0 projection in Python
    (no SelectNode needed for a width-1 stream).
    2-D events (multi-value rows): tees the stream into N per-column iterators
    and drives an N-input combine_latest + select Pipeline.  The N tee'd iterators
    are advanced in lockstep by _LazyDag, so the tee buffer is at most O(N)
    elements (constant in stream length - O(1) per event).

    Positional feeds (index=None) are converted to row-number indices
    internally and the None index is restored on output.
    """
    import itertools
    from .dag import Input, Pipeline

    cols, scalar = _normalize_columns(columns)

    try:
        head = next(events)
    except StopIteration:
        return

    head_val, head_idx = head
    all_events = itertools.chain([head], events)
    N = np.atleast_1d(np.asarray(head_val, dtype=np.float64)).size
    positional = head_idx is None

    # Validate column range against the first event's width.
    for c in cols:
        if c >= N:
            raise ValueError(
                f"select: column {c} out of range for width {N}")

    if N == 1:
        # 1-D: width-1 passthrough; SelectNode adds no value here.
        for v, k in all_events:
            val = float(np.atleast_1d(np.asarray(v, dtype=np.float64))[0])
            if scalar:
                yield val, k
            else:
                yield [val], k
        return

    # 2-D: convert positional None indices to row numbers before feeding the Pipeline.
    if positional:
        def _add_rownum(evs):
            for i, (v, _k) in enumerate(evs):
                yield v, i
        working_events = _add_rownum(all_events)
    else:
        working_events = all_events

    # Tee N copies (lockstep); each copy is consumed once per event.
    tees = itertools.tee(working_events, N)

    def _col(teed, col):
        for row_val, k in teed:
            r = np.atleast_1d(np.asarray(row_val, dtype=np.float64))
            yield float(r[col]), k

    col_iters = [_col(tees[i], i) for i in range(N)]
    col_inputs = [Input(f"c{i}") for i in range(N)]
    columns_arg = cols[0] if scalar else cols
    dag = Pipeline(col_inputs, [select(combine_latest(*col_inputs), columns=columns_arg)])
    inner = dag(*col_iters)

    if positional:
        for v, _k in inner:
            yield v, None
    else:
        yield from inner


def select(values, columns, index=None):
    """Pick column(s) from a wide (M, N) value stream.

    columns is an int (result is 1-D) or a sequence of ints (result is 2-D with
    those columns in order). Row count and index are unchanged (row-preserving
    shape op, not cardinality). Column indices must be in range and non-negative.
    Values-first + polymorphic: raw arrays, (values, index) tuple, or graph Node.
    index=None means positional (no index allocation). Returns (values, index) /
    Node.
    When ``values`` is a ``(values, index)`` tuple it carries its own index;
    ``index=`` applies only to raw arrays.
    """
    if is_node(values):
        return make_operator_node(Select, (values,), {"columns": columns})
    if _is_lazy_stream(values):
        return _select_lazy_cpp(values, columns)
    vals, idx = _as_vi(values, index)
    vals = np.asarray(vals, dtype=np.float64)
    cols, scalar = _normalize_columns(columns)
    # Validate columns in Python before building the Pipeline (preserves exact
    # ValueError message regardless of which regime is active).
    width = 1 if vals.ndim == 1 else vals.shape[1]
    for c in cols:
        if c >= width:
            raise ValueError(
                f"select: column {c} out of range for width {width}")
    picked, out_idx = _select_via_cpp(vals, idx, cols, scalar)
    return picked, out_idx   # index is unchanged (row-preserving)


_RESAMPLE_AGGS = ("first", "last", "min", "max", "sum", "count", "mean", "ohlc",
                  "ohlcv", "ohlcv2")
_RESAMPLE_FILLS = ("skip", "nan", "carry")

# Finance aliases: map short OHLC names to their canonical engine strings.
_FINANCE_ALIASES = {
    "high":  "max",
    "low":   "min",
    "open":  "first",
    "close": "last",
}

# Statistical synonyms: map short stat names to the Expanding* functors that
# implement them (the engine has no built-in code for these).
_STAT_SYNONYMS = {   # string synonym -> Expanding* functor class name
    "std": "ExpandingStd", "var": "ExpandingVar", "prod": "ExpandingProd",
    "skew": "ExpandingSkew", "kurt": "ExpandingKurt",
}

# Full set of accepted agg string names for error messages.
_ALL_AGG_NAMES = _RESAMPLE_AGGS + tuple(_FINANCE_ALIASES) + tuple(_STAT_SYNONYMS)


def _resolve_agg(agg):
    """Normalise an agg alias or synonym to the form the engine/functor path expects.

    - Finance aliases ('high', 'low', 'open', 'close') become their canonical
      engine strings ('max', 'min', 'first', 'last') and continue on the fast
      engine path.
    - Statistical synonyms ('std', 'var', 'prod', 'skew', 'kurt') become the
      corresponding Expanding* functor instances and route through the functor-agg
      path that already exists.
    - Everything else (canonical engine strings, arbitrary functors, dicts) is
      returned unchanged.
    """
    if not isinstance(agg, str):
        return agg
    if agg in _FINANCE_ALIASES:
        return _FINANCE_ALIASES[agg]
    if agg in _STAT_SYNONYMS:
        import screamer   # deferred to avoid circular import at module load
        # instantiate only the requested reducer (a fresh instance per call, since
        # reducers are stateful); do not build the four we would discard
        return getattr(screamer, _STAT_SYNONYMS[agg])()
    return agg


_OFFSET_UNIT_MAP = {   # offset unit -> numpy timedelta64 unit
    "s":   "s",
    "min": "m",
    "T":   "m",
    "h":   "h",
    "D":   "D",
}


def _parse_offset_string(s):
    """Parse an offset string like '5min', '1h', 'D' into a np.timedelta64.

    Supported units: s, min (alias T), h, D.
    A bare unit without a multiplier means multiplier 1 (e.g. 'min' == '1min').
    Raises ValueError for an unrecognised or calendar-based (M, Y) unit.
    """
    m = re.fullmatch(r"(\d+)?([a-zA-Z]+)", s.strip())
    if m is None:
        raise ValueError(
            f"resample: cannot parse offset string {s!r}; "
            "expected '<N><unit>', e.g. '5min', '1h', '30s'")
    mult_str, unit = m.group(1), m.group(2)
    mult = int(mult_str) if mult_str else 1
    if unit not in _OFFSET_UNIT_MAP:
        raise ValueError(
            f"resample: unsupported offset unit {unit!r} in {s!r}; "
            "supported: s, min, T, h, D "
            "(calendar offsets such as month/year are not supported)")
    return np.timedelta64(mult, _OFFSET_UNIT_MAP[unit])


def _resample_datetime_freq(freq, index):
    """Convert freq (offset string, timedelta, or np.timedelta64) to an integer
    span in the index's own datetime64 resolution units.

    Returns ("span", width_int64).
    Raises TypeError for int/float freq on a datetime64 index.
    Raises ValueError for unsupported/calendar offset units or a non-exact conversion.
    """
    # Reject numeric freq types on a datetime64 index.
    # Note: np.timedelta64 inherits from np.integer, so exclude it explicitly.
    _is_number = (isinstance(freq, (int, float, np.integer, np.floating))
                  and not isinstance(freq, np.timedelta64))
    if _is_number:
        raise TypeError(
            f"resample: freq={freq!r} is a number but the index is datetime64; "
            "pass an offset string (e.g. '1min') or a datetime.timedelta / "
            "np.timedelta64 for a datetime64 index")

    # Convert freq to np.timedelta64
    if isinstance(freq, str):
        td = _parse_offset_string(freq)
    elif isinstance(freq, datetime.timedelta):
        # Use exact integer arithmetic: timedelta stores days, seconds, microseconds.
        total_us = (freq.days * 86_400 * 1_000_000
                    + freq.seconds * 1_000_000
                    + freq.microseconds)
        td = np.timedelta64(total_us, "us")
    elif isinstance(freq, np.timedelta64):
        td = freq
        unit = np.datetime_data(td)[0]
        if unit in ("M", "Y"):
            raise ValueError(
                f"resample: offset {freq!r} uses a calendar unit ('{unit}') "
                "which has variable length and cannot map to a fixed span; "
                "use D, h, min, or s instead")
    else:
        raise TypeError(
            f"resample: for a datetime64 index, freq must be a string offset, "
            f"datetime.timedelta, or np.timedelta64; got {type(freq).__name__!r}")

    # Determine the index's datetime64 resolution unit (e.g. 'ns', 's', 'ms', 'us')
    idx = np.asarray(index)
    res_unit = np.datetime_data(idx.dtype)[0]

    # Compute the span width as a float, then verify it is an exact integer.
    try:
        td_in_res = td / np.timedelta64(1, res_unit)
    except TypeError as exc:
        raise ValueError(
            f"resample: cannot convert offset {freq!r} to index resolution "
            f"'{res_unit}': {exc}") from exc

    width_int = int(td_in_res)
    if width_int != td_in_res:
        raise ValueError(
            f"resample: offset {freq!r} = {td} is not an exact multiple of the "
            f"index resolution '{res_unit}' (result {td_in_res} is not an integer); "
            "choose an offset that divides evenly into the index resolution units")
    if width_int <= 0:
        raise ValueError(
            f"resample: offset {freq!r} produces a non-positive span ({width_int}); "
            "freq must be positive")

    return ("span", width_int)


def _resample_freq_to_engine(freq, index):
    """Translate freq into (mode, width) for the engine. freq is always a WINDOW
    (a span over the index); Option B: freq=window, count=arrival.

    integer freq     -> width = int(freq); mode "span" (a span in index units).
                        For a positional/no-index stream the index is the row
                        number, so a span of W coincides with count=W; for a Node
                        or lazy stream the span is resolved against the runtime
                        index (which is why freq cannot be forced to count here).
    datetime64 index -> offset/timedelta -> int64 units via _resample_datetime_freq.
    Raises on a non-positive width or a nonsensical (index dtype, freq type) pair.
    """
    if index is not None and np.asarray(index).dtype.kind == "M":   # datetime64
        return _resample_datetime_freq(freq, index)
    if isinstance(freq, (str, np.timedelta64)) or hasattr(freq, "total_seconds"):
        # offset string, np.timedelta64, or datetime.timedelta on a non-datetime64
        # index - not supported (an integer index expects an integer span)
        raise TypeError(
            f"resample: freq={freq!r} is an offset/timedelta but the index is not "
            "datetime64; pass an integer span for an integer index, or use a "
            "datetime64 index with an offset string / timedelta")
    width = int(freq)
    if width <= 0:
        raise ValueError("resample: freq must be a positive integer")
    return "span", width


def _resample_via_cpp(feed, *, every, count, agg, origin, label, fill="skip"):
    """Run resample on the C++ engine via a one-node Pipeline, for batch OR lazy input.

    Builds the minimal ``Input -> Resample`` graph the Node regime already uses,
    then defers to ``dag(feed)``. Rule A on the Pipeline decides the mode: a concrete
    ``(vals, idx)`` pair runs batch and returns ``(out_values, out_index)``; a lazy
    iterator of ``(value, index)`` events returns a lazy iterator of
    ``(bar_value, bar_label)``. No Python windowing - all bucketing and NaN-ignore
    accumulation happens in the C++ core.
    """
    from .dag import Input, Pipeline
    src = Input("x")
    node = resample(src, every=every, count=count, agg=agg,
                    origin=origin, label=label, fill=fill)
    dag = Pipeline([src], [node])
    return dag(feed)


def _resample_validate(freq, every, count, agg, label, fill="skip"):
    # Exactly one of freq, every, count must be provided.
    given = sum(x is not None for x in [freq, every, count])
    if given != 1:
        raise ValueError("resample: pass exactly one of freq=, every=, or count=")
    # Positivity guard: every=0 would reach the engine's floordiv(_, 0) -> a hard
    # SIGFPE crash; count<1 never completes a bucket. Reject both up front (this
    # runs before the Node dispatch, so it guards the graph path too).
    # freq positivity is checked inside _resample_freq_to_engine.
    if every is not None and int(every) <= 0:
        raise ValueError("resample: every must be positive")
    if count is not None and int(count) < 1:
        raise ValueError("resample: count must be >= 1")
    # agg may be a builtin string or an arbitrary functor reducer (an EvalOp).
    # dict agg is no longer supported; raise immediately with a migration hint.
    if isinstance(agg, dict):
        raise ValueError(
            "resample no longer accepts agg={...}; build columns with "
            "combine_latest of per-stat resamples, e.g. "
            "combine_latest(resample(price, freq=..., agg='first'), "
            "resample(vol, freq=..., agg='sum')); see docs")
    if isinstance(agg, str) and agg not in _RESAMPLE_AGGS:
        raise ValueError(
            f"resample: unknown agg string {agg!r}; "
            f"valid names are {sorted(_ALL_AGG_NAMES)}")
    if label not in ("left", "right"):
        raise ValueError('resample: label must be "left" or "right"')
    if fill not in _RESAMPLE_FILLS:
        raise ValueError(f"resample: fill must be one of {_RESAMPLE_FILLS}")



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
    Python only splits columns, orchestrates the sub-calls, and stacks.

    Column order: open(0), high(1), low(2), close(3)[, volume(4)[, buy_vol(4),
    sell_vol(5)]].
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
        return stacked, out_idx

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
    return stacked, out_idx


def resample(values, index=None, *, freq=None, every=None, count=None, agg="last",
             origin=0, label="left", fill="skip"):
    """Causal windowed downsample of a 1-D value stream.

    Pass exactly one of ``freq``, ``every``, or ``count`` to bound the bars.
    ``freq`` is a **window** (a span over the index) and the recommended form: it
    is always equivalent to ``every`` in every regime (batch, Node, and lazy),
    resolved against the runtime index. For a positional (no-index) stream the
    index is the row number, so a span of W coincides with ``count=W``. ``every``
    is the same window as ``freq`` (retained for now); ``count`` is the arrival
    form. ``every`` and ``count`` are the explicit forms:

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
      ``(open, high, low, close)``.
    * **functor**: any screamer functor used as a reducer (e.g.
      ``ExpandingSkew()``).  The functor is ``reset()`` at each bar boundary
      and fed every in-bar sample; its last output before the close is emitted.
      A single-output functor returns a 1-D result; a multi-output functor
      returns a 2-D result.  The functor must accept exactly
      1 input (single-value stream); a multi-input functor raises at runtime.
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

    Values-first + polymorphic: raw arrays, (values, index) tuple, or graph Node.
    The returned index is always the bar labels (a real array, never None) - even
    for a positional (no-index) input, which resamples by row position.

    Graph form: resample(node, ...) where node is a Node.
    When ``values`` is a ``(values, index)`` tuple it carries its own index;
    ``index=`` applies only to raw arrays.
    """
    agg = _resolve_agg(agg)
    _resample_validate(freq, every, count, agg, label, fill)
    # Translate freq= into every=/count= using the index context.  After this
    # block every and count follow the existing internal convention so all
    # downstream code is unchanged.
    if freq is not None:
        if _is_vi_pair(values):
            _idx_ctx = values[1]     # (values, index) tuple carries its own index
        else:
            _idx_ctx = index            # raw array, Node, or lazy stream
        _mode, _width = _resample_freq_to_engine(freq, _idx_ctx)
        every = _width if _mode == "span" else None
        count = _width if _mode == "count" else None
    if is_node(values):
        if agg in ("ohlcv", "ohlcv2"):
            raise ValueError(
                f"resample: the agg='{agg}' string shorthand is eager-only and is "
                "not supported in the graph (Node) regime. In a graph, build "
                "multi-column bars with combine_latest of per-stat resample nodes, "
                "e.g. combine_latest(resample(price, every=W, agg='first'), "
                "resample(vol, every=W, agg='sum')).")
        return make_operator_node(Resample, (values,), {
            "every": every, "count": count, "agg": agg,
            "origin": origin, "label": label, "fill": fill})
    if _is_lazy_stream(values):
        # Rule A: a lazy iterator of (value, index) events -> a lazy iterator of
        # (bar_value, bar_label). Drive the same C++ resample node as batch through
        # the Stage-2 lazy Pipeline; no Python windowing accumulator runs here.
        if agg in ("ohlcv", "ohlcv2"):
            raise ValueError(
                "resample(<iterator>) supports string and functor scalar aggs "
                "only; ohlcv/ohlcv2 aggs are eager-only. Materialize the "
                "stream to an array for those.")
        return _resample_via_cpp(values, every=every, count=count, agg=agg,
                                 origin=origin, label=label, fill=fill)
    vals, idx_in = _as_vi(values, index)
    vals = np.asarray(vals, dtype=np.float64)
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
    if idx_in is None:
        idx = np.arange(len(vals), dtype=np.int64)
    else:
        idx = np.asarray(idx_in)

    # ohlcv / ohlcv2: orchestrate existing C++ reducers over a 2-column input.
    if agg in ("ohlcv", "ohlcv2"):
        return _resample_ohlcv(vals, idx, agg, every=every, count=count,
                               origin=origin, label=label, fill=fill)

    # Delegate all bucketing/accumulation to the C++ engine.
    out_v, out_idx = _resample_via_cpp(
        (vals, idx), every=every, count=count, agg=agg, origin=origin, label=label,
        fill=fill)
    # Always return a (values, index) tuple; bar labels are always real (never None).
    return out_v, out_idx



# ---------------------------------------------------------------------------
# CamelCase config-first classes (public API, step 3E).
# The lowercase functions below remain as transitional shims during migration
# and are removed in the final 3E task.
# ---------------------------------------------------------------------------


class Merge:
    """Index-sorted N-way merge.  Config-first form of :func:`merge`.

    ``Merge()(*values, index=None)`` is equivalent to
    ``merge(*values, index=None)``.  No configuration at construction time.
    """

    def __call__(self, *values, index=None):
        return merge(*values, index=index)


class CombineLatest:
    """As-of latest-value join.  Config-first form of :func:`combine_latest`.

    ``CombineLatest(emit="when_all")(*values, index=None)`` is equivalent to
    ``combine_latest(*values, index=None, emit="when_all")``.

    ``func=`` is not available on the class surface; apply a functor to the
    aligned output instead, e.g.
    ``Sub()(CombineLatest()(a, b))``.
    """

    def __init__(self, emit="when_all"):
        self._emit = emit

    def __call__(self, *values, index=None):
        return combine_latest(*values, index=index, emit=self._emit)


class Dropna:
    """Drop NaN events.  Config-first form of :func:`dropna`.

    ``Dropna(how="any")(values, index=None)`` is equivalent to
    ``dropna(values, index, how="any")``.
    """

    def __init__(self, how="any"):
        self._how = how

    def __call__(self, values, index=None):
        return dropna(values, index, how=self._how)


class Select:
    """Pick columns from a wide stream.  Config-first form of :func:`select`.

    ``Select(columns)(values, index=None)`` is equivalent to
    ``select(values, columns, index=None)``.
    """

    def __init__(self, columns):
        self._columns = columns

    def __call__(self, values, index=None):
        return select(values, self._columns, index)


class Resample:
    """Causal windowed downsample.  Config-first form of :func:`resample`.

    ``Resample(freq=None, count=None, ...)(values, index=None)`` is equivalent
    to ``resample(values, index, freq=freq, count=count, ...)``.

    ``every=`` is not available on the class surface (Option B: freq/count
    only).  Use the :func:`resample` function directly if you need ``every=``.
    """

    def __init__(self, freq=None, count=None, agg="last", origin=0,
                 label="left", fill="skip"):
        self._cfg = dict(freq=freq, count=count, agg=agg, origin=origin,
                         label=label, fill=fill)

    def __call__(self, values, index=None):
        return resample(values, index, **self._cfg)


# transitional: the public API for the five operators above is their CamelCase
# class. The lowercase functions remain as the shared implementation and as
# backward-compatible shims; they are removed in the final 3E task.
