"""Streaming operators (merge, combine_latest, dropna, Filter, select,
split, resample).

Each operator dispatches on input type (Rule A): concrete data runs eager;
a lazy iterator of (value, index) events returns a lazy iterator. Builds and
runs C++ node graphs; dtype detection here chooses the int64 or float64
index-type instantiation, and the per-event work is all C++.
"""
import datetime
import math
import re

import numpy as np

from . import screamer_bindings as _b
from .dag import is_node, make_operator_node, _is_vi_pair, _as_float64, _refuse_text, _setting

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
    "Delay",
]


def _is_lazy_stream(x):
    """Rule A: a lazy stream is an iterator (has ``__next__``) that is NOT a
    concrete container (list, tuple, or ndarray). Generators and
    ``iter(...)`` qualify; concrete data does not."""
    return hasattr(x, "__next__") and not isinstance(x, (list, tuple, np.ndarray))


def _as_vi(x, idx, who=""):
    """Normalize a data input to (values_ndarray, index_or_None).

    Accepts:
    - a bare array (positional or with explicit idx) -> (np.asarray(x), idx)
    - a (values, index) 2-tuple -> pass through
    """
    if _is_vi_pair(x):
        return x[0], x[1]
    if not hasattr(x, "dtype") and not isinstance(x, (list, tuple)):
        _refuse_text(x, who)   # a time or a ctypes object is not values
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


def _to_streams(inputs, index, who=""):
    """Normalize each input to a (values, index) tuple. `index` is None (all
    positional) or a list aligned with inputs (per-stream index array or None)."""
    if index is not None and len(index) != len(inputs):
        raise ValueError("index list length must match the number of streams")
    return [_as_vi(x, None if index is None else index[i], who)
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


def from_pandas(obj, layout=None):
    """Build a (values, index) tuple from a pandas Series or DataFrame.

    data -> values (numpy array), pandas index -> index array. Without
    ``layout`` the values are ``obj.to_numpy()`` with the frame's own dtype and
    columns. ``layout`` names the axes as read and reshapes the float64 values
    like :func:`reshape`: ``"TA"`` for one column per asset, or ``"TAF"`` for a
    DataFrame whose two-level columns hold every ``(asset, field)`` pair, which
    gives ``(T, A, F)``. pandas always materializes an index (a RangeIndex for
    a default one), so a positional stream does not round-trip to index=None
    through to_pandas/from_pandas. Any other object raises TypeError.
    """
    mod = getattr(type(obj), "__module__", None)
    if not (isinstance(mod, str) and mod.split(".")[0] == "pandas"
            and hasattr(obj, "to_numpy") and hasattr(obj, "index")):
        name = type(obj).__name__
        article = "an" if name[:1].lower() in "aeiou" else "a"
        raise TypeError(f"from_pandas: pass a pandas Series or DataFrame, got {article} {name}")
    if layout is None:
        return np.asarray(obj.to_numpy()), np.asarray(obj.index)
    from .shape import frame_axes, reshape_array
    return reshape_array(frame_axes(obj)[0], layout), np.asarray(obj.index)


def _run_chain(functors, values, index=None, return_index=False):
    """Run source -> functors[0] -> ... -> collector in batch.

    Returns values array by default. Pass return_index=True to get
    (out_index, out_values) instead.

    index=None uses the row number (int64) as the ordering index.
    """
    values = _as_float64(values, contiguous=True)
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
    Raises ValueError if streams is empty, TypeError for numbers against times
    or two time units. Int and float indexes mix as in the iterator form.
    """
    if not streams:
        raise ValueError(f"{who}: needs at least one stream")
    # One index kind and unit across the streams, as the iterator forms (and a
    # Pipeline call) require: plain numbers, or times of one unit.
    _b._check_index_kinds([idx for idx, _ in streams])
    kinds, norm_index, norm_vals = [], [], []
    for idx, values in streams:
        kind, k = _index_dtype_kind(idx)
        kinds.append(kind)
        norm_index.append(k)
        norm_vals.append(_as_float64(values, contiguous=True, who=who))
    if len(set(kinds)) == 1:
        return kinds[0], norm_index, norm_vals
    # Int and float indexes are both plain numbers (the one-kind check above
    # refused times). As in the iterator form, the first non-empty index sets
    # the kind: float reads every index as float64; int needs every float
    # index to be whole (the engine's index rule raises TypeError otherwise).
    kind = next((kd for kd, k in zip(kinds, norm_index) if k.size), kinds[0])
    if kind == "f64":
        norm_index = [np.ascontiguousarray(k, dtype=np.float64) for k in norm_index]
    else:
        for i, (kd, k) in enumerate(zip(kinds, norm_index)):
            if kd == "i64":
                continue
            try:
                norm_index[i] = _b._index_i64(k)
            except TypeError as e:
                raise TypeError(f"{who}: the index of input {i}: {e}") from None
    return kind, norm_index, norm_vals


def _streams_to_indexed(streams, who):
    """(kind, index_list, vals_list, positional). Uniform positional or indexed;
    mixing positional and indexed raises. Positional streams have no index list
    (None): C++ pairs them by position and checks their lengths."""
    indexed = [s[1] is not None for s in streams]
    if any(indexed) and not all(indexed):
        raise ValueError(
            f"{who}: cannot align positional and indexed streams; give every "
            "stream an index, or none")
    vals = [_as_float64(s[0], contiguous=True, who=who) for s in streams]
    if not any(indexed):
        # Paired by position in C++, along the T axis each layout names.
        return "i64", None, vals, True
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
    streams = _to_streams(values, index, who)   # (values, index) passthrough; raw wrapped with index[i]
    for k, (v, _) in enumerate(streams):
        # A number is one event, not a stream (as in JavaScript).
        if np.ndim(v) == 0 and not hasattr(v, "__next__"):
            raise TypeError(f"{who}: input {k} is a number, expected an array, a "
                            "(values, index) pair or an iterator")
    has_idx = [s[1] is not None for s in streams]
    if any(has_idx) and not all(has_idx):
        raise ValueError(
            f"{who}: cannot mix positional and indexed streams; give every "
            "stream an index, or none")
    vals = [_as_float64(s[0], contiguous=True, who=who) for s in streams]
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



def _combine_latest_lazy(values, emit, layout=None):
    """Dispatch gate for lazy combine_latest: classify sources on first next().

    Classification (and peeking one head per source) is deferred to the first
    call to ``__next__``, so construction consumes nothing. The first event of
    each source fixes its fields and asset axes, as in the eager path; the C++
    join plan (``include/screamer/streams/join_plan.h``) reads the events and
    assembles the rows. Positional sources are paired by position in C++
    (``_CombineZip``): each step takes one event from every source, so they
    must stay in step and end together. Indexed ((value, index)) sources take
    the as-of join in C++ (``_CombineLatestLazy``): the merge iterator's order
    and the batch join's rows, on an int64 or a float64 index as the first
    event's index is (as the batch form's first index picks its kind).
    """
    import itertools
    positional, sources, kind = _classify_lazy_sources(values, "combine_latest")
    heads = []
    for k, src in enumerate(sources):
        h = next(src, _EMPTY)
        heads.append(h)
        if h is not _EMPTY:
            sources[k] = itertools.chain([h], src)
    if all(h is _EMPTY for h in heads):
        return
    plan = _b._JoinPlan([None if h is _EMPTY else
                         (1,) + np.shape(_as_float64(h if positional else h[0], who="combine_latest"))
                         for h in heads], layout)
    if positional:
        yield from _b._CombineZip(sources, plan, _as_float64)
        return
    cls = _b._CombineLatestLazy_f64 if kind == "f64" else _b._CombineLatestLazy_i64
    yield from cls(sources, plan, emit == "when_all")


def combine_latest(*values, index=None, emit="when_all", layout=None, max_pending=None):
    """As-of latest-value join of N streams: one row per distinct index (same-index
    events coalesce). Values-first + polymorphic: raw arrays, (values, index)
    tuples, or graph Nodes.

    No-index (positional) inputs are paired by position: each step takes one
    value from every input, so they must stay in step and end together; returns
    (aligned_values, None). Indexed inputs perform an as-of join aligned on each
    stream's index, returning (aligned_values, index). Node inputs return a Node.

    emit="when_all" (default) suppresses output until every input has a value;
    emit="on_any" emits from the first event (NaN for unseen inputs). To transform
    each aligned row, apply a functor to the output, e.g. Sub()(combine_latest(a, b)).
    When a (values, index) tuple is passed it carries its own index; index=
    applies only to raw arrays.

    ``layout`` names each input's axes (one string per input, or one for all),
    as in :func:`reshape`; inputs with asset axes broadcast them and concatenate
    their fields. By default a 1-D or 2-D input is fields and a 3-D input is
    ``(T, A, F)``.

    ``max_pending`` (Node inputs only) is the join's own cap on the events it
    holds while it waits for another input; unset, the Pipeline's.
    """
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    if any(is_node(v) for v in values):
        if layout is not None:
            raise ValueError(
                "CombineLatest(layout=...) applies to arrays and iterators; "
                "streams inside a Pipeline carry fields only")
        params = {"emit": emit}
        if max_pending is not None:
            params["max_pending"] = _b._max_pending(max_pending, "CombineLatest: max_pending")
        return make_operator_node(CombineLatest, values, params)
    if max_pending is not None:
        raise TypeError("CombineLatest: max_pending applies to Node inputs, the join "
                        "inside a Pipeline")
    if values and all(_is_lazy_stream(v) for v in values):
        return _combine_latest_lazy(values, emit, layout)
    if any(_is_lazy_stream(v) for v in values):
        raise TypeError(
            "combine_latest: cannot mix lazy iterator and concrete inputs; pass "
            "all generators or all arrays")
    streams = _to_streams(values, index, "combine_latest")
    kind, idx, vals, positional = _streams_to_indexed(streams, "combine_latest")
    # The layout join, the as-of join and the assembly all run in C++
    # (include/screamer/streams/join_plan.h).
    fn = _b._combine_latest_f64 if kind == "f64" else _b._combine_latest_i64
    out_index, aligned = fn(idx, vals, emit == "when_all", layout)
    return aligned, (None if positional else out_index)



def _positional_rows(events):
    """The (value, None) events of a positional lazy source with their row
    numbers as index, as a batch call numbers its rows."""
    for i, (v, _k) in enumerate(events):
        yield v, i


def _lazy_heads(sources):
    """Peeks the first (value, index) event of each lazy source. Returns
    (heads, sources) with each source re-chained with its head; an empty
    source has the head _EMPTY."""
    import itertools
    heads, chained = [], []
    for src in sources:
        it = iter(src)
        head = next(it, _EMPTY)
        heads.append(head)
        chained.append(it if head is _EMPTY else itertools.chain([head], it))
    return heads, chained


def _run_lazy_pipeline(dag, sources, who):
    """Runs a lazy one-node Pipeline over (value, index) sources, where index
    None marks a positional source (rows numbered as in a batch call). Every
    source must be positional, or none. Yields (value, index) events, the index
    None again for positional sources. A width-1 event (a one-element array or
    list) reads as a number, as before."""
    heads, sources = _lazy_heads(sources)
    found = [h for h in heads if h is not _EMPTY]
    if not found:
        return
    positional = {h[1] is None for h in found}
    if len(positional) > 1:
        raise ValueError(f"{who}: cannot mix positional and indexed streams; give "
                         "every stream an index, or none")
    positional = positional.pop()
    if positional:
        sources = [_positional_rows(src) for src in sources]
    head_val = found[0][0]
    if not isinstance(head_val, (float, int)) and head_val is not None \
            and np.size(_as_float64(head_val, who=who)) == 1:
        sources = [((float(np.ravel(_as_float64(v, who=who))[0]), k) for v, k in src)
                   for src in sources]
    for v, k in dag(*sources):
        yield v, (None if positional else k)


def _one_node_pipeline(n, build, who):
    """A Pipeline of n inputs feeding one stream node, build(*inputs)."""
    from .dag import Input, Pipeline
    ins = [Input(f"_{who}{i}") for i in range(n)]
    dag = Pipeline(ins, [build(*ins)])
    dag._who = who
    return dag


def _run_eager_pipeline(dag, streams, who):
    """Runs a one-node Pipeline on (values, index) streams, index None for a
    positional stream (every stream positional, or none). Returns (values,
    index), the index None for positional streams."""
    has_idx = [k is not None for _, k in streams]
    if any(has_idx) and not all(has_idx):
        raise ValueError(f"{who}: cannot mix positional and indexed streams; give "
                         "every stream an index, or none")
    positional = not any(has_idx)
    feeds = [v if positional else (v, k) for v, k in streams]
    out_vals, out_idx = dag(*feeds)
    return out_vals, (None if positional else out_idx)


def dropna(*values, index=None, how="any"):
    """Drop events whose value is NaN. Each input may be 1-D (M,) or 2-D (M, N).

    how="any" (default) drops a row if any component is NaN; how="all" only if
    all are. Values-first + polymorphic: raw arrays, (values, index) tuples,
    iterators of (value, index) events, or graph Nodes. Several inputs are
    several streams: their events are merged in index order (ties in input
    order) and each is dropped on its own, as the Dropna node of a Pipeline does.
    index=None means positional (no index allocation). Returns (values, index) /
    Node restricted to the surviving rows. Surviving values are returned
    as float64. Causal and cardinality-changing: batch and streaming drop the
    same rows.
    An index is passed as a ``(values, index)`` tuple, or with ``index=`` for a
    single raw array.
    """
    if how not in ("any", "all"):
        raise ValueError('dropna: how must be "any" or "all"')
    if not values:
        raise ValueError("dropna: needs at least one stream")
    if any(is_node(v) for v in values):
        return make_operator_node(Dropna, values, {"how": how})
    build = lambda *ins: dropna(*ins, how=how)
    if all(_is_lazy_stream(v) for v in values):
        if index is not None:
            raise TypeError("dropna: index= applies to arrays; an iterator carries "
                            "its index in each (value, index) event")
        return _run_lazy_pipeline(_one_node_pipeline(len(values), build, "dropna"),
                                  values, "dropna")
    if any(_is_lazy_stream(v) for v in values):
        raise TypeError("dropna: cannot mix lazy iterator and concrete inputs; pass "
                        "all generators or all arrays")
    if index is not None and len(values) > 1:
        raise TypeError("dropna: index= applies to one input; pass each of several "
                        "inputs as a (values, index) tuple")
    streams = []
    for v in values:
        vals, idx = _as_vi(v, index, "dropna")
        streams.append((_as_float64(vals, contiguous=True, who="dropna"), idx))
    return _run_eager_pipeline(_one_node_pipeline(len(values), build, "dropna"),
                               streams, "dropna")


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
        dag._who = "Filter"
        return dag(data, mask)


def split(values, sources, index=None, n=None):
    """Partition a merged tagged stream back into per-source streams.

    The inverse of merge. ``values`` may be a raw value array or a
    ``(values, index)`` tuple (which carries its own index); ``sources`` is
    always passed separately.
    Returns a list of ``(values, index)`` pairs. Positional (index=None) uses
    ``None`` for the per-source index.
    ``n`` sets how many output streams to produce (default: max(sources)+1); pass
    it explicitly to include sources that emitted nothing. A source is a whole
    number (a bool, a fractional or a non-finite source raises TypeError) from
    0 to 2147483646 (else ValueError), and an ``n`` at or below the largest
    source raises ValueError. ``values`` may have any rank with time first:
    each event is a row, and each output keeps the row shape.
    """
    # Argument adaptation only: the rules (shapes, the source rule, n) and the
    # partition are C++ (include/screamer/streams/split.h), shared with the
    # JavaScript split.
    if _is_vi_pair(values):
        index = values[1]          # the tuple carries its own index
        values = values[0]
    values = _as_float64(values, contiguous=True, who="split")
    # A list or tuple is read item by item in C++, and a native integer, bool
    # or float64 array as it is, so the kind of each source is checked before
    # any conversion (a bool is a wrong kind). Anything else is read as
    # float64 (a string is a TypeError, None is NaN).
    if not isinstance(sources, (list, tuple)):
        if not hasattr(sources, "dtype"):
            sources = np.asarray(sources)
        dt = sources.dtype
        if dt.kind == "O" and np.ndim(sources) == 1:
            sources = list(sources)
        elif getattr(dt, "isnative", False) and (dt.kind in "biu" or dt == np.float64):
            sources = np.ascontiguousarray(sources)
        else:
            sources = _as_float64(sources, contiguous=True)
    if n is not None:
        _setting("split: n", n, True)
    if index is not None:
        if getattr(getattr(index, "dtype", None), "kind", "O") in ("U", "S", "O"):
            _b._refuse_text(index if hasattr(index, "dtype") else np.asarray(index))
        _b._check_index_kinds([index])        # numbers or times, as every stream index
        index = _index_dtype_kind(index)[1]
    return _b._split(values, sources, index, n)


def _normalize_columns(columns):
    """Validate columns (int or non-negative int sequence) -> (list_of_ints, scalar).

    Returns (cols, is_scalar). is_scalar is True when a bare int was given (the
    eager result is then 1-D). Negative indices are rejected explicitly.
    """
    scalar = np.ndim(columns) == 0
    cols = ([int(_setting("select: columns", columns, True))] if scalar
            else [int(_setting("select: columns", c, True)) for c in columns])
    for c in cols:
        if c < 0:
            raise ValueError(f"select: column index must be non-negative, got {c}")
    return cols, scalar


def _select_lazy_cpp(events, columns):
    """Lazy select: a width-1 stream passes through (column 0) in Python; a
    wide stream runs the C++ SelectNode through a one-input lazy Pipeline.

    Generator function: no events are consumed until the first next() call.
    The first event fixes the width; positional feeds (index None) are
    numbered by row internally and get the None index back on output.
    """
    cols, scalar = _normalize_columns(columns)
    heads, (events,) = _lazy_heads([events])
    head = heads[0]
    if head is _EMPTY:
        return
    N = np.size(_as_float64(head[0], who="select"))
    for c in cols:
        if c >= N:
            raise ValueError(
                f"select: column {c} out of range for width {N}")
    if N == 1:
        # Width 1: column 0 is the value itself.
        for v, k in events:
            val = float(np.ravel(_as_float64(v, who="select"))[0])
            yield (val if scalar else [val]), k
        return
    columns_arg = cols[0] if scalar else cols
    dag = _one_node_pipeline(1, lambda x: select(x, columns=columns_arg), "select")
    yield from _run_lazy_pipeline(dag, [events], "select")


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
    vals, idx = _as_vi(values, index, "select")
    vals = _as_float64(vals, contiguous=True, who="select")
    cols, scalar = _normalize_columns(columns)
    # Validate columns in Python before building the Pipeline (preserves exact
    # ValueError message regardless of which regime is active).
    width = 1 if vals.ndim == 1 else vals.shape[1]
    for c in cols:
        if c >= width:
            raise ValueError(
                f"select: column {c} out of range for width {width}")
    if vals.ndim == 1:
        # Width 1: column 0 is the values; a new array, never the input's.
        return (vals.copy() if scalar else vals.reshape(-1, 1).copy()), idx
    # A wide stream runs the C++ SelectNode through a one-input Pipeline,
    # which keeps every row (equal indexes too).
    columns_arg = cols[0] if scalar else cols
    dag = _one_node_pipeline(1, lambda x: select(x, columns=columns_arg), "select")
    return _run_eager_pipeline(dag, [(vals, idx)], "select")


_RESAMPLE_AGGS = ("first", "last", "min", "max", "sum", "count", "mean", "ohlc",
                  "ohlcv", "ohlcv2", "ohlc_bars", "ohlcv_bars")
_RESAMPLE_FILLS = ("skip", "nan", "carry")

# Bar aggs that require multi-column (2-D) input.
_BAR_AGGS_2COL = frozenset(("ohlcv", "ohlcv2"))       # exactly 2 columns
_BAR_AGGS_4COL = frozenset(("ohlc_bars",))             # exactly 4 columns
_BAR_AGGS_NCOL = frozenset(("ohlcv_bars",))            # 5+ columns (checked in C++)
_COLUMN_AGGS = frozenset(("first", "last", "min", "max", "sum", "count", "mean", "ohlc"))


def _column_shape(shape):
    """The (*A) part of a (T, *A) or (T, *A, 1) value array (rule 1)."""
    trailing = tuple(shape[1:])
    return trailing[:-1] if len(trailing) >= 2 and trailing[-1] == 1 else trailing

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
    if not isinstance(freq, str):
        _setting("resample: freq", freq, True)           # a bool is not a number
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


def _resample_via_cpp(feed, *, every, count, threshold, agg, origin, label,
                      fill="skip", clock=False, max_fill=None):
    """Run resample on the C++ engine via a one-node Pipeline, for batch OR lazy input.

    Builds the minimal ``Input -> Resample`` graph the Node regime already uses,
    then defers to ``dag(feed)``. Rule A on the Pipeline decides the mode: a concrete
    ``(vals, idx)`` pair runs batch and returns ``(out_values, out_index)``; a lazy
    iterator of ``(value, index)`` events returns a lazy iterator of
    ``(bar_value, bar_label)``. No Python windowing - all bucketing and NaN-ignore
    accumulation happens in the C++ core.

    For ByCumulative mode (threshold is not None): feed is a (vals, idx) pair where
    vals is 2-D (N, 2) with columns [value, driver]. The single Input node receives
    the combined frame; the ResampleNode sees width-2 frames and uses ByCumulative
    logic internally.

    ByClock mode (clock=True) is NOT routed through this function; it uses
    _resample_clock_batch / _resample_clock_lazy with a 2-input Pipeline instead.
    """
    from .dag import Input, Pipeline
    src = Input("x")
    node = resample(src, every=every, count=count, threshold=threshold, agg=agg,
                    origin=origin, label=label, fill=fill, clock=clock,
                    max_fill=max_fill)
    dag = Pipeline([src], [node])
    dag._who = "resample"
    return dag(feed)


def _resample_clock_batch(vals, val_idx, clk_vals, clk_idx, *, agg, fill,
                           origin, label, max_age):
    """Batch ByClock resample: run a 2-input Pipeline (value + clock) with the
    C++ ResampleNode(ByClock) in 2-port mode.

    Port 0 receives value events; port 1 receives clock events. Events from both
    ports arrive in global index order (MergeSource in run_batch guarantees this).
    The node accumulates value events and emits at each clock tick.
    """
    from .dag import Input, Pipeline
    val_in = Input("v")
    clk_in = Input("c")
    node = resample(val_in, clk_in, clock=True, agg=agg, fill=fill,
                    origin=origin, label=label, max_age=max_age)
    dag = Pipeline([val_in, clk_in], [node])
    dag._who = "resample"
    # The indexes go to the Pipeline as given; the engine applies the index rule.
    vals_f = _as_float64(vals, contiguous=True, who="resample")
    clk_f = np.ascontiguousarray(clk_vals if clk_vals.ndim == 1 else clk_vals[:, 0],
                                  dtype=np.float64)
    return dag((vals_f, val_idx), (clk_f, clk_idx))


def _lazy_resample_rows(dag, iters, agg):
    """The C++ lazy driver of a one-output resample Pipeline. A builtin column
    agg gives each bar in its event's shape, (*A,) or (*A, 4) for ohlc, as a
    batch row; the driver reads and shapes the events, no Python per event."""
    from . import screamer_bindings as _b
    if isinstance(agg, str) and agg in _COLUMN_AGGS:
        return _b._LazyDriver(dag._cg, iters, 1, "Pipeline", 0,
                              [4] if agg == "ohlc" else [])
    return _b._LazyDriver(dag._cg, iters, 1)


def _resample_clock_lazy(values, clock_stream, *, agg, fill, origin, label,
                           max_age):
    """Lazy ByClock resample: run a 2-input Pipeline (value + clock) in lazy mode.

    Builds the same 2-input Pipeline as _resample_clock_batch but feeds it with
    lazy iterators. The C++ _LazyDriver merges the two input iterators by index
    and routes value events to port 0 and clock events to port 1 of the
    ResampleNode(ByClock), producing a lazy output iterator. Array events are
    resampled per column and each bar comes back in the event's shape.
    """
    from .dag import Input, Pipeline
    val_in = Input("v")
    clk_in = Input("c")
    node = resample(val_in, clk_in, clock=True, agg=agg, fill=fill,
                    origin=origin, label=label, max_age=max_age)
    dag = Pipeline([val_in, clk_in], [node])
    return _lazy_resample_rows(dag, [iter(values), iter(clock_stream)], agg)


def _resample_validate(freq, every, count, threshold, agg, label, fill="skip",
                       clock=False, max_age=None):
    if max_age is not None:
        if not clock or agg != "last" or fill != "carry":
            raise ValueError(
                "resample: max_age requires clock=True, agg='last', and fill='carry'")
        if (isinstance(max_age, (bool, np.bool_)) or
                not isinstance(max_age, (int, np.integer))):
            raise TypeError(
                "resample: max_age must be a non-negative integer in index units")
        if max_age < 0:
            raise ValueError("resample: max_age must be a non-negative integer")
    for name, x in (("threshold", threshold), ("every", every), ("count", count)):
        _setting(f"resample: {name}", x, name != "threshold")
    # Exactly one of freq, every, count, threshold, clock must be provided.
    given = sum(x is not None for x in [freq, every, count, threshold])
    if clock:
        given += 1
    if given != 1:
        raise ValueError(
            "resample: pass exactly one of freq=, every=, count=, threshold=, or clock=")
    # threshold positivity guard.
    if threshold is not None and float(threshold) <= 0.0:
        raise ValueError("resample: threshold must be > 0")
    # threshold= feeds only a width-1 value to the node (the driver occupies col 1
    # of the combined width-2 frame). Multi-column-value aggs (ohlcv, ohlcv2,
    # ohlc_bars, ohlcv_bars) require input_col >= 1 which is never wired, so the
    # volume column silently emits 0. Reject these combos with a clear message.
    _THRESHOLD_MULTICOL_BLOCKED = frozenset(("ohlcv", "ohlcv2", "ohlc_bars", "ohlcv_bars"))
    if threshold is not None and isinstance(agg, str) and agg in _THRESHOLD_MULTICOL_BLOCKED:
        raise ValueError(
            f"resample threshold= mode supports single-column value aggs "
            f"(first/last/min/max/sum/count/mean/ohlc); "
            f"agg={agg!r} requires multi-column value input that is not "
            f"available in threshold= mode. "
            f"For OHLCV volume bars, resample the OHLC and the volume "
            f"separately and combine: e.g. "
            f"combine_latest(Resample(price_vi, vol, threshold=T, agg='ohlc'), "
            f"Resample(price_vi, vol, threshold=T, agg='sum'))."
        )
    # Positivity guard: every=0 would reach the engine's floordiv(_, 0) -> a hard
    # SIGFPE crash; count<1 never completes a bucket. Reject both up front (this
    # runs before the Node dispatch, so it guards the graph path too).
    # freq positivity is checked inside _resample_freq_to_engine.
    if every is not None and int(every) <= 0:
        raise ValueError("resample: every must be positive")
    if count is not None and int(count) < 1:
        raise ValueError("resample: count must be >= 1")
    # clock= and threshold= only support builtin string aggs. A functor/EvalOp
    # reducer is routed to GenericResampleNode which ignores the clock/cumulative
    # mode, producing silently wrong output. Reject early with a clear message.
    if not isinstance(agg, str) and not isinstance(agg, dict):
        if clock:
            raise ValueError(
                "resample clock= mode requires a builtin string agg "
                "(first/last/min/max/sum/count/mean/ohlc); "
                "functor reducers are not supported with clock=. "
                "Apply your functor to the resampled output instead.")
        if threshold is not None:
            raise ValueError(
                "resample threshold= mode requires a builtin string agg "
                "(first/last/min/max/sum/count/mean/ohlc); "
                "functor reducers are not supported with threshold=. "
                "Apply your functor to the resampled output instead.")
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



def _resample_lazy(values, *, every, count, agg, origin, label, fill, max_fill=None):
    """Lazy resample; events whose value is an array of assets are resampled per
    column and come back in the same shape (the C++ driver reads and shapes
    them)."""
    if not isinstance(agg, str):
        return _resample_lazy_functor(values, every=every, count=count, agg=agg,
                                      origin=origin, label=label, fill=fill,
                                      max_fill=max_fill)
    from .dag import Input, Pipeline
    src = Input("x")
    dag = Pipeline([src], [resample(src, every=every, count=count, agg=agg,
                                    origin=origin, label=label, fill=fill,
                                    max_fill=max_fill)])
    return _lazy_resample_rows(dag, [iter(values)], agg)


def _resample_lazy_functor(values, *, every, count, agg, origin, label, fill,
                           max_fill=None):
    """Lazy resample with a functor agg, which aggregates one column."""
    import itertools
    it = iter(values)
    head = next(it, _EMPTY)
    if head is _EMPTY:
        return
    v0 = _as_float64(head[0], who="resample")
    if v0.ndim > 0:
        raise ValueError(
            f"resample: a functor agg aggregates one column; got events of shape {v0.shape}. "
            "Resample each column with its own functor, or use a builtin agg.")
    yield from _resample_via_cpp(itertools.chain([head], it), every=every, count=count,
                                 threshold=None, agg=agg, origin=origin, label=label,
                                 fill=fill, max_fill=max_fill)


def resample(values, driver=None, index=None, *, freq=None, every=None, count=None,
             threshold=None, clock=False, agg="last", origin=0, label="left",
             fill="skip", max_age=None, max_fill=None):
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
    one or more buckets have no samples). It applies under ``freq=`` (``every=``)
    and ``clock=``; a ``count=`` or ``threshold=`` bar is never empty, so a
    ``fill`` other than ``"skip"`` raises ``ValueError`` there.

    * ``"skip"`` (default): no row for an empty bucket (legacy behavior).
    * ``"nan"``: an all-NaN row at each empty bucket's label.
    * ``"carry"``: each column takes its last non-NaN value: an empty bucket
      repeats those values, and a NaN cell of a bar takes its column's value
      (a column without a value yet stays NaN). Sum- and volume-like columns
      are carried like the others.

    Only internal gaps are filled. Leading buckets before the first event and
    trailing buckets after the last are not synthesized here (output starts at the
    first tick's bucket; trailing empties need the clock/``advance()`` mechanism).
    ``"carry"`` works per column, for every column alike (so an empty bar
    carries a count/sum column's prior value, not 0; use ``"nan"`` if that is
    wrong for a column), and skips a bucket that has no prior emitted row.
    A filled bucket emits a synthetic row without feeding or resetting the reducer,
    so a functor reducer starts each real bar clean. The trailing partial bucket is
    still emitted once at end of input. Integer index-space.

    ``max_fill`` bounds the empty bars that one gap may fill: a fill resample
    whose event (or ``advance(now)``) would close more empty buckets at once
    raises ``ValueError`` before it builds any bar. The default is 1,000,000;
    pass an integer from 1 to 2**53 to change it.

    Values-first + polymorphic: raw arrays, (values, index) tuple, or graph Node.
    The returned index is always the bar labels (a real array, never None) - even
    for a positional (no-index) input, which resamples by row position.

    Graph form: resample(node, ...) where node is a Node.
    When ``values`` is a ``(values, index)`` tuple it carries its own index;
    ``index=`` applies only to raw arrays.
    """
    agg = _resolve_agg(agg)
    _resample_validate(freq, every, count, threshold, agg, label, fill, clock,
                       max_age)
    if max_age is not None:
        max_age = int(max_age)
    if max_fill is not None:
        max_fill = _b._max_fill(max_fill)
    # Only a set max_fill enters a graph node, so a default graph's to_dict
    # is unchanged.
    fill_kw = {} if max_fill is None else {"max_fill": max_fill}
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

    # ByClock mode: values is the value stream, driver is the clock stream.
    # At each clock event the current bucket is closed and emitted.
    if clock:
        if driver is None:
            raise ValueError(
                "resample: clock=True requires a clock stream as the second argument")
        if is_node(values):
            if not is_node(driver):
                raise ValueError(
                    "resample: clock= requires a clock Node as the second argument")
            # Graph regime: wire two separate Input nodes into the 2-port
            # ResampleNode(ByClock). The compiled graph routes port(0) <- value
            # and port(1) <- clock.
            return make_operator_node(Resample, (values, driver), {
                "clock": True, "agg": agg,
                "every": None, "count": None,
                "origin": origin, "label": label, "fill": fill,
                "max_age": max_age, **fill_kw})
        # Eager / lazy path: merge value and clock events into a tagged stream
        # [value..., is_clock] where is_clock=0 for value events, 1 for clock.
        # The merged stream is fed to the C++ ResampleNode(ByClock) which
        # inspects the last column to route each event.
        if _is_lazy_stream(values) or _is_lazy_stream(driver):
            return _resample_clock_lazy(values, driver, agg=agg, fill=fill,
                                        origin=origin, label=label,
                                        max_age=max_age)
        vals, val_idx = _as_vi(values, index, "resample")
        vals = _as_float64(vals, who="resample")
        if max_age is not None and vals.ndim != 1:
            raise ValueError("resample: max_age requires a 1-D value stream")
        clk_vals, clk_idx = _as_vi(driver, None, "resample")
        clk_vals = _as_float64(clk_vals, who="resample")
        if val_idx is None:
            val_idx = np.arange(len(vals), dtype=np.int64)
        if clk_idx is None:
            clk_idx = np.arange(len(clk_vals), dtype=np.int64)
        # A builtin agg on (T, *A) reduces each column, as with freq= and
        # count=: the result is (T', *A), with a last axis of 4 for ohlc.
        columns = None
        if vals.ndim > 1 and agg in _COLUMN_AGGS:
            columns = _column_shape(vals.shape)
            vals = vals.reshape(len(vals), math.prod(columns))
        out_v, out_idx = _resample_clock_batch(
            vals, np.asarray(val_idx), clk_vals, np.asarray(clk_idx),
            agg=agg, fill=fill, origin=origin, label=label, max_age=max_age)
        if columns is not None:
            out_v = np.asarray(out_v).reshape(
                (len(out_idx),) + columns + ((4,) if agg == "ohlc" else ()))
        return out_v, out_idx

    # Guard: a second positional argument (driver) without a mode selector is ambiguous
    # when driver is a stream, Node, or (values, index) pair. Plain numpy arrays are
    # silently ignored (legacy: the second positional was once treated as an index).
    if driver is not None and threshold is None and not clock:
        driver_is_plain_array = isinstance(driver, np.ndarray)
        if not driver_is_plain_array:
            raise ValueError(
                "resample: a second positional argument (driver/clock stream) requires "
                "either threshold= or clock=True to specify the mode")

    # ByCumulative mode: values is the value stream, driver is the driver stream.
    # The node expects a width-2 input [value, driver] combined on the same index.
    if threshold is not None:
        if is_node(values):
            if driver is not None and is_node(driver):
                # Graph regime: Resample(value_node, driver_node, threshold=T, agg=...)
                # The C++ graph builder joins value and driver into one
                # [value, driver] row (CombineLatest, when_all) and resamples it.
                return make_operator_node(Resample, (values, driver), {
                    "threshold": threshold, "agg": agg,
                    "every": None, "count": None,
                    "origin": origin, "label": label, "fill": fill, **fill_kw})
            elif driver is None:
                # Internal path: called from _resample_via_cpp with a single Input
                # that already carries the combined width-2 frame (value+driver).
                # Create the ResampleNode(ByCumulative) directly on the single input.
                return make_operator_node(Resample, (values,), {
                    "threshold": threshold, "agg": agg,
                    "every": None, "count": None,
                    "origin": origin, "label": label, "fill": fill, **fill_kw})
            else:
                raise ValueError(
                    "resample: threshold= requires a driver Node as the second argument")
        # Eager / lazy path: combine value and driver into a width-2 frame.
        if driver is None:
            raise ValueError(
                "resample: threshold= requires a driver stream as the second argument")
        vals, idx_in = _as_vi(values, index, "resample")
        vals = _as_float64(vals, who="resample")
        if vals.ndim != 1:
            raise ValueError(
                "resample: threshold= requires a 1-D value stream; "
                f"got shape {vals.shape}")
        drv = _as_float64(driver, who="resample")
        if drv.ndim != 1:
            raise ValueError(
                "resample: threshold= requires a 1-D driver stream; "
                f"got shape {drv.shape}")
        if len(drv) != len(vals):
            raise ValueError(
                "resample: value and driver must have the same length; "
                f"got {len(vals)} and {len(drv)}")
        if idx_in is None:
            idx = np.arange(len(vals), dtype=np.int64)
        else:
            idx = np.asarray(idx_in)
        # Combine value + driver into a (N, 2) array: col0=value, col1=driver.
        combined_vals = np.ascontiguousarray(
            np.column_stack([vals, drv]), dtype=np.float64)
        out_v, out_idx = _resample_via_cpp(
            (combined_vals, idx), every=None, count=None, threshold=threshold,
            agg=agg, origin=origin, label=label, fill=fill, max_fill=max_fill)
        return out_v, out_idx

    if is_node(values):
        return make_operator_node(Resample, (values,), {
            "every": every, "count": count, "agg": agg,
            "origin": origin, "label": label, "fill": fill, **fill_kw})
    if _is_lazy_stream(values):
        # Rule A: a lazy iterator of (value, index) events -> a lazy iterator of
        # (bar_value, bar_label). Drive the same C++ resample node as batch through
        # the Stage-2 lazy Pipeline; no Python windowing accumulator runs here.
        return _resample_lazy(values, every=every, count=count, agg=agg,
                              origin=origin, label=label, fill=fill, max_fill=max_fill)
    vals, idx_in = _as_vi(values, index, "resample")
    vals = _as_float64(vals, who="resample")
    # Multi-column bar aggs require specific input widths.
    if agg in _BAR_AGGS_2COL:
        if vals.ndim != 2 or vals.shape[1] != 2:
            col_desc = "volume" if agg == "ohlcv" else "signed_volume"
            raise ValueError(
                f"resample: agg='{agg}' requires exactly 2 columns "
                f"[price, {col_desc}]; "
                f"got shape {vals.shape}. Pass np.column_stack([price, {col_desc}]).")
    elif agg in _BAR_AGGS_4COL:
        if vals.ndim != 2 or vals.shape[1] != 4:
            raise ValueError(
                f"resample: agg='{agg}' requires exactly 4 columns "
                f"[open, high, low, close]; got shape {vals.shape}. "
                "Pass np.column_stack([open_arr, high_arr, low_arr, close_arr]).")
    elif agg in _BAR_AGGS_NCOL:
        # The column count (at least 5) is the engine's rule, shared with the
        # Pipeline node (resample_params.h check_ohlcv_bars_width).
        if vals.ndim != 2:
            raise ValueError(
                f"resample: agg='{agg}' requires a 2-D input "
                f"[open, high, low, close, vol1, ...]; got shape {vals.shape}.")
    elif vals.ndim != 1 and not (isinstance(agg, str) and agg in _COLUMN_AGGS):
        raise ValueError(
            f"resample: a functor agg aggregates one column; got shape {vals.shape}. "
            "Resample each column with its own functor, or use a builtin agg.")
    columns = None
    if vals.ndim > 1 and isinstance(agg, str) and agg in _COLUMN_AGGS:
        columns = _column_shape(vals.shape)
        vals = vals.reshape(len(vals), math.prod(columns))

    # Use explicit index or row positions when positional.
    if idx_in is None:
        idx = np.arange(len(vals), dtype=np.int64)
    else:
        idx = np.asarray(idx_in)

    # Delegate all bucketing/accumulation to the C++ engine.
    # Multi-column bar aggs (ohlcv, ohlcv2, ohlc_bars, ohlcv_bars) use the plan
    # path inside _resample_via_cpp via the compiled DAG which reads the plan
    # from _BAR_AGG_FIXED_PLANS or the dynamic OhlcvBars sentinel in C++.
    out_v, out_idx = _resample_via_cpp(
        (vals, idx), every=every, count=count, threshold=None, agg=agg,
        origin=origin, label=label, fill=fill, max_fill=max_fill)
    if columns is not None:
        out_v = np.asarray(out_v).reshape(
            (len(out_idx),) + columns + ((4,) if agg == "ohlc" else ()))
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

    ``CombineLatest(emit="when_all", layout=None, max_pending=None)(*values, index=None)`` is
    equivalent to ``combine_latest(*values, index=None, emit=..., layout=...)``.

    ``func=`` is not available on the class surface; apply a functor to the
    aligned output instead, e.g.
    ``Sub()(CombineLatest()(a, b))``.
    """

    def __init__(self, emit="when_all", layout=None, max_pending=None):
        if max_pending is not None:      # a setting raises at construction
            max_pending = _b._max_pending(max_pending, "CombineLatest: max_pending")
        self._emit = emit
        self._layout = layout
        self._max_pending = max_pending

    def __call__(self, *values, index=None):
        return combine_latest(*values, index=index, emit=self._emit, layout=self._layout,
                              max_pending=self._max_pending)


class Dropna:
    """Drop NaN events.  Config-first form of :func:`dropna`.

    ``Dropna(how="any")(*values, index=None)`` is equivalent to
    ``dropna(*values, index=index, how="any")``. Each positional argument is
    one stream; an index goes in a ``(values, index)`` tuple (or ``index=``
    for one raw array).
    """

    def __init__(self, how="any"):
        self._how = how

    def __call__(self, *values, index=None):
        return dropna(*values, index=index, how=self._how)


class Select:
    """Pick columns from a wide stream.  Config-first form of :func:`select`.

    ``Select(columns)(values, index=None)`` is equivalent to
    ``select(values, columns, index=None)``.
    """

    def __init__(self, columns):
        _normalize_columns(columns)          # a string or negative column raises here
        self._columns = columns

    def __call__(self, values, index=None):
        return select(values, self._columns, index)


class Resample:
    """Causal windowed downsample.  Config-first form of :func:`resample`.

    Classic config-first form::

        Resample(freq=5, agg='last')(values, index)
        Resample(count=N, agg='ohlc')(values, index)

    Cumulative-driver (information bar) form::

        Resample(values, driver, threshold=T, agg='ohlc')

    Clock-mode (target-clock resample / as-of) form::

        Resample(values, clock_stream, clock=True, agg='last', fill='carry')

    The cumulative-driver and clock-mode forms are direct-call shorthands that
    return ``(out_values, out_index)`` immediately.

    ``every=`` is not available on the class surface (Option B: freq/count
    only).  Use the :func:`resample` function directly if you need ``every=``.
    """

    def __new__(cls, values_or_freq=None, driver=None, *,
                freq=None, every=None, count=None, threshold=None, clock=False,
                agg="last", origin=0, label="left", fill="skip", max_age=None,
                max_fill=None):
        # Detect whether the first positional arg is data (cumulative-driver or
        # clock-mode shorthand form) or a config param (classic config-first form).
        # When data IS provided, compute and return the result directly.
        _is_data = (values_or_freq is not None and
                    not isinstance(values_or_freq, (int, float,
                                                    np.integer, np.floating)))
        if _is_data and (threshold is not None or clock or every is not None):
            # Direct call: Resample(value, driver, threshold=T, ...) or
            #              Resample(value, clock, clock=True, ...) or
            #              Resample(value, every=W, ...)
            # Returns the resample result (a tuple), not a Resample instance.
            return resample(values_or_freq, driver,
                            freq=freq, every=every, count=count, threshold=threshold,
                            clock=clock, agg=agg, origin=origin, label=label,
                            fill=fill, max_age=max_age, max_fill=max_fill)
        if _is_data and driver is not None:
            # A driver (second positional arg) with no mode selector: route through
            # resample() which will raise the "requires threshold= or clock=True" error.
            return resample(values_or_freq, driver,
                            freq=freq, every=every, count=count, threshold=threshold,
                            clock=clock, agg=agg, origin=origin, label=label,
                            fill=fill, max_age=max_age, max_fill=max_fill)
        # Classic config-first: return a Resample instance.
        return super().__new__(cls)

    def __init__(self, values_or_freq=None, driver=None, *,
                 freq=None, every=None, count=None, threshold=None, clock=False,
                 agg="last", origin=0, label="left", fill="skip", max_age=None,
                 max_fill=None):
        # Skip init when __new__ already returned a non-Resample result.
        if not isinstance(self, Resample):
            return
        # Classic config-first form: values_or_freq is a numeric freq or None.
        eff_freq = (values_or_freq
                    if (values_or_freq is not None and
                        isinstance(values_or_freq, (int, float,
                                                    np.integer, np.floating)))
                    else freq)
        if max_fill is not None:               # a setting raises at construction
            max_fill = _b._max_fill(max_fill)
        self._cfg = dict(freq=eff_freq, every=every, count=count, threshold=threshold,
                         clock=clock, agg=agg, origin=origin, label=label, fill=fill,
                         max_age=max_age, max_fill=max_fill)

    def __call__(self, values, index=None):
        # Classic config-first form: Resample(cfg)(values, index=None)
        # For ByCumulative mode (threshold=) or ByClock mode (clock=True),
        # the caller passes values as a (values, index) tuple and driver=None
        # (the driver/clock is stored in the node spec built by the graph path).
        return resample(values, None, index, **self._cfg)


def delay(values, index=None, *, duration):
    """Re-stamp each event's index by `duration` (index units); values unchanged.

    Requires an explicit index (duration is time-based). Returns (values, index +
    duration) as a concrete stream, or a graph Node when `values` is a Node."""
    duration = int(_setting("delay: duration", duration, True))
    if is_node(values):
        return make_operator_node(Delay, (values,), {"duration": duration})
    values, index = _as_vi(values, index, "delay")
    if index is None:
        raise TypeError(
            "Delay requires an explicit index (duration is time-based); "
            "call Delay(duration)(values, index).")
    return _delay_via_cpp((values, index), duration=duration)


def _delay_via_cpp(feed, *, duration):
    from .dag import Input, Pipeline
    src = Input("x")
    node = delay(src, duration=duration)
    dag = Pipeline([src], [node])
    dag._who = "delay"
    return dag(feed)


class Delay:
    """Index re-stamp operator. Config-first form: ``Delay(duration)(values, index)``.

    Shifts each event's index by `duration` (in index units) while leaving values
    unchanged. Stateless, 1:1, lossless. Requires an explicit index.
    """

    __name__ = "Delay"

    def __init__(self, duration):
        self._duration = int(_setting("Delay: duration", duration, True))

    def __call__(self, values, index=None):
        return delay(values, index, duration=self._duration)


# transitional: the public API for the five operators above is their CamelCase
# class. The lowercase functions remain as the shared implementation and as
# backward-compatible shims; they are removed in the final 3E task.
