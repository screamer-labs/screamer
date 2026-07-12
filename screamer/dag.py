"""Computational DAG definition: symbolic Node handles (DAG-1)."""

import numpy as np
from collections import deque

__all__ = ["Node", "Input", "Pipeline"]

_RESAMPLE_AGG_CODE = {"first": 0, "last": 1, "min": 2, "max": 3,
                      "sum": 4, "count": 5, "mean": 6, "ohlc": 7}
_RESAMPLE_FILL_CODE = {"skip": 0, "nan": 1, "carry": 2}


class Node:
    """An immutable handle for a stream inside a pipeline.

    You do not usually construct a Node directly: ``Input(name)`` returns one, and
    applying a functor or a stream operator to a Node returns another. ``op``
    records the operation (an input, a functor instance, or an operator) and
    ``inputs`` is a tuple of upstream Nodes.
    """
    __slots__ = ("op", "inputs")
    is_node = True

    def __init__(self, op, inputs=()):
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "inputs", tuple(inputs))

    def __setattr__(self, *a):
        raise AttributeError("Node is immutable")

    def __repr__(self):
        if isinstance(self.op, tuple) and self.op and self.op[0] == "input":
            return f"Input({self.op[1]!r})"
        if isinstance(self.op, tuple) and self.op and self.op[0] == "operator":
            name = self.op[1].__name__
        else:
            name = type(self.op).__name__
        return f"Node({name}, {len(self.inputs)} input(s))"


def is_node(obj):
    return getattr(obj, "is_node", False) is True


def Input(name):
    """Create a source Node - a named placeholder for a timed stream."""
    return Node(("input", name))


def make_functor_node(functor, args):
    """Build a Node for a functor applied to Node args. Called by the C++ hook."""
    inputs = tuple(args)
    for n in inputs:
        if not is_node(n):
            raise TypeError("all arguments must be Nodes when building a graph")
    return Node(functor, inputs)


def make_operator_node(fn, node_args, kwargs):
    """Build a Node for a stream operator applied to Node args."""
    return Node(("operator", fn, kwargs), tuple(node_args))


def _int64_index(idx):
    """Coerce an index array to contiguous int64, raising on a fractional value.

    The engine is int64-indexed and float indices are not yet supported, so a
    non-integer index is rejected loudly rather than silently truncated (a floored
    index would change bucketing and alignment versus the caller's intent).
    Integer-valued floats (2.0) are lossless and accepted. Integer dtypes skip the
    check entirely, so the common path adds no scan.
    """
    arr = np.asarray(idx)
    if arr.dtype.kind == "f":
        out = arr.astype(np.int64)
        if not np.array_equal(out, arr):
            raise TypeError(
                "index must be integer-valued; a fractional index was seen. The "
                "streaming engine is int64-indexed (float indices are not yet "
                "supported). Pass integer indices.")
        return np.ascontiguousarray(out)
    return np.ascontiguousarray(arr, dtype=np.int64)


def _is_vi_pair(x):
    """The canonical concrete-stream detection: a (values, index) 2-tuple whose
    first element is an ndarray. Defined once so every site that may receive a
    bare array or a (values, index) pair applies the same rule."""
    return isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], np.ndarray)


def _as_stream(feed):
    """Normalize a feed to (index_array, values_array) for the compiled engine.

    Accepts:
    - bare value array -> positional (index = row-number int64 via np.arange)
    - (values, index) pair -> values-first user convention; flipped for engine
      (index None -> row-number)
    """
    if _is_vi_pair(feed):
        values, index = feed   # user provides (values, index) - values-first
        values_arr = np.ascontiguousarray(values, dtype=np.float64)
        if index is None:
            return (np.arange(values_arr.shape[0], dtype=np.int64), values_arr)
        return (_int64_index(index), values_arr)
    values = np.asarray(feed, dtype=np.float64)
    return (np.arange(values.shape[0], dtype=np.int64), values)


def _align_results(results, align_outputs):
    """Return (values, index) for M==1; tuple of (values, index) pairs otherwise.

    align_outputs=False: independent per-output (values, index) pairs.
    align_outputs=True:  co-indexed via combine_latest (coalesces - one row per
    distinct index), all pairs share the same index array.

    Input ``results`` is a list of (index_array, values_array) pairs as returned
    by the compiled engine.
    """
    if len(results) == 1:
        k, v = results[0]
        return (v, k)   # values-first
    if not align_outputs:
        return tuple((v, k) for k, v in results)   # values-first
    from .streams import combine_latest
    # combine_latest accepts (values, index) tuples; coalesces to one row per index
    vi_list = [(v, k) for k, v in results]
    aligned, aligned_index = combine_latest(*vi_list, emit="when_all")
    return tuple((aligned[:, j], aligned_index) for j in range(len(results)))  # values-first


def _reachable_inputs(outputs):
    """Return the set of Input Nodes reachable from the output nodes."""
    seen, stack, inputs = set(), list(outputs), []
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node.op, tuple) and node.op and node.op[0] == "input":
            inputs.append(node)
        else:
            stack.extend(node.inputs)
    return inputs


def _check_stateful_safety(outputs):
    """A functor instance may back at most one node."""
    seen, stack, used = set(), list(outputs), {}
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        op = node.op
        if not isinstance(op, tuple):                 # a functor instance
            if id(op) in used:
                raise ValueError(
                    "the same functor instance backs two nodes; construct a "
                    "fresh functor per node (state cannot be shared); "
                    f"offending node: {node!r}")
            used[id(op)] = node
        stack.extend(node.inputs)


class _LiveDag:
    """A live streaming session over a compiled Pipeline. `input` in push() is an Input
    name (str) or its positional index (int)."""
    def __init__(self, dag):
        self._dag = dag
        self._cg = dag._cg
        # name -> signature index, built once so per-event push() stays O(1)
        self._src_of = {nm: i for i, nm in enumerate(dag._input_order)}
        self._cg.reset()

    def push(self, input, index, value):
        src = input if isinstance(input, int) else self._src_of[input]
        self._cg.push_event(int(src), int(index), float(value))
        return self

    def advance(self, now):
        """Close every window whose boundary has passed by logical time `now`."""
        self._cg.advance(int(now))
        return self

    def flush(self):
        """Finalize the current partial window(s) now (end-of-loop / on demand)."""
        self._cg.flush()
        return self

    def result(self):
        """Aligned outputs accumulated so far (drains the buffers)."""
        results = self._cg.drain()
        results = [(k, v.reshape(-1) if v.shape[1] == 1 else v) for (k, v) in results]
        return _align_results(results, self._dag.align_outputs)


_NOT_PULLED = object()   # sentinel: iterator head has not been fetched yet


class _LazyDag:
    """Lazy pull driver: run the compiled graph event by event over input iterators.

    Each feed is an iterator of (value, index) events. Events are merged by index
    (as-of, ascending) across inputs, pushed one at a time, and the outputs that
    closed after each push are yielded. On exhaustion the graph is flushed and the
    trailing rows are yielded. Values match the batch dag(...) result (the oracle).

    Heads are fetched lazily: nothing is consumed from any feed until the first
    call to __next__, so the driver is non-eager on construction.
    """
    def __init__(self, dag, feeds):
        self._dag = dag
        self._cg = dag._cg
        # The lazy multi-output path coalesces outputs at a shared index (when_all),
        # matching align_outputs=True. The independent-per-output (align_outputs=False)
        # multi-output case is not yet handled lazily; fail fast rather than misalign.
        if not dag.align_outputs and len(dag.outputs) > 1:
            raise NotImplementedError(
                "lazy dag(iterables) with align_outputs=False and multiple outputs "
                "is not yet supported; use the batch call dag(arrays) for now")
        self._cg.reset()
        # one (index, value) iterator per input, in signature order
        self._iters = [iter(feeds[nm]) for nm in dag._input_order]
        # _heads[i] is _NOT_PULLED (not yet fetched), (idx, val) (ready), or None (done)
        self._heads = [_NOT_PULLED] * len(self._iters)
        self._pending = deque()         # buffered output rows not yet yielded
        self._done = False
        # Watermark as-of join state for the multi-output when_all path. The M
        # outputs drain at independent rates (one may race ahead of another), so
        # their merged index stream is NOT globally sorted; we cannot forward-fill
        # naively. Instead buffer drained events and only finalize an index once
        # every output has drained strictly past it (its as-of value is then
        # settled). Mirrors combine_latest(emit="when_all") + collapse-per-index.
        self._latest = [None] * len(dag.outputs)   # each output's as-of value
        self._wm = [None] * len(dag.outputs)       # highest index drained per output
        self._buf = []                             # (index, out_pos, val), unsettled

    @staticmethod
    def _pull(it):
        try:
            v, k = next(it)
            ik = int(k)
            if ik != k:
                raise TypeError(
                    "index must be integer-valued; a fractional index was seen. "
                    "The streaming engine is int64-indexed (float indices are not "
                    "yet supported). Pass integer indices.")
            return (ik, float(v))
        except StopIteration:
            return None

    def __iter__(self):
        return self

    def _drain_rows(self):
        drained = self._cg.drain()      # list of (idx_arr, vals_2d) per output
        if len(drained) == 1:
            idx_arr, vals = drained[0]
            for k, row in zip(idx_arr, vals):
                yield (float(row[0]) if vals.shape[1] == 1 else tuple(map(float, row)), int(k))
            return
        # multi-output with align_outputs=True: buffer this drain's events and
        # advance the per-output watermarks, then settle every index the join can
        # now finalize.
        for out_pos, (idx_arr, vals) in enumerate(drained):
            scalar = vals.shape[1] == 1
            for k, row in zip(idx_arr, vals):
                k = int(k)
                val = float(row[0]) if scalar else tuple(map(float, row))
                self._buf.append((k, out_pos, val))
                if self._wm[out_pos] is None or k > self._wm[out_pos]:
                    self._wm[out_pos] = k
        if any(w is None for w in self._wm):
            return                       # some output has not fired; nothing settled
        # safe frontier: no output can still emit an index below min(watermark),
        # so every index strictly less than it is final. Emit those now.
        yield from self._settle(min(self._wm))

    def _settle(self, bound):
        """Emit one row per distinct buffered index < ``bound`` (all buffered when
        ``bound`` is None), forward-filling each output's as-of value. Suppress
        until every output has a value (when_all)."""
        keep, ready = [], []             # one partition pass over the buffer
        for e in self._buf:
            (ready if (bound is None or e[0] < bound) else keep).append(e)
        if not ready:
            return
        self._buf = keep
        ready.sort(key=lambda e: e[0])   # stable: last drained value wins per index
        i, n = 0, len(ready)
        while i < n:
            k = ready[i][0]
            while i < n and ready[i][0] == k:
                self._latest[ready[i][1]] = ready[i][2]
                i += 1
            if all(v is not None for v in self._latest):
                yield tuple(self._latest) + (k,)

    def __next__(self):
        while True:
            if self._pending:
                return self._pending.popleft()
            if self._done:
                raise StopIteration
            # lazily fill any heads that have not been fetched yet
            for i, h in enumerate(self._heads):
                if h is _NOT_PULLED:
                    self._heads[i] = self._pull(self._iters[i])
            # pick the input with the smallest next index (as-of merge)
            nxt = min((h[0] for h in self._heads if h is not None), default=None)
            if nxt is None:
                # all inputs exhausted: flush trailing events, settle whatever the
                # watermark join was still holding back, then stop
                self._cg.flush()
                self._pending.extend(self._drain_rows())
                self._pending.extend(self._settle(None))
                self._done = True
                continue
            # push every input whose head is at this index (same-index coalescing)
            for i, h in enumerate(self._heads):
                if h is not None and h[0] == nxt:
                    self._cg.push_event(i, nxt, h[1])
                    self._heads[i] = _NOT_PULLED    # mark for lazy refill
            self._pending.extend(self._drain_rows())


class Pipeline:
    """A reusable N-in / M-out function you define once and call on stored or live data.

    Arguments:

    - ``inputs``: ordered list of ``Input(...)`` nodes that define the call
      signature. Feeds are bound positionally, or by name via a keyword call.
    - ``outputs``: ordered list of output nodes to evaluate.
    - ``align_outputs`` (default ``True``): when ``True``, co-index all M outputs
      onto a shared, sorted index axis, so each output carries its as-of value at
      every unique union index (combine_latest's per-event intermediate rows are
      collapsed to one row per index) and the returned (values, index) pairs have
      equal length. When ``False``, return independent per-output streams whose
      lengths may differ.

    Call ``pipe(*feeds)`` (positional) or ``pipe(**named_feeds)`` (by Input name)
    to run the pipeline. Each feed may be a bare value array (positional, index
    = row-number), or a ``(values, index)`` pair (values-first).
    Pass generators of ``(value, index)`` pairs to run the pipeline lazily, event
    by event, with byte-identical results (the lazy pull path).
    Returns a single ``(values, index)`` pair when M == 1, or a tuple of pairs
    when M > 1.
    """

    def __init__(self, inputs, outputs, align_outputs=True):
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.align_outputs = align_outputs
        for n in self.inputs:
            if not is_node(n) or not (isinstance(n.op, tuple) and n.op[0] == "input"):
                raise ValueError("every entry in inputs must be an Input(...) node")
        for n in self.outputs:
            if not is_node(n):
                raise ValueError("every entry in outputs must be a Node")
        reachable = _reachable_inputs(self.outputs)
        reachable_ids = {id(n) for n in reachable}
        declared_ids = {id(n) for n in self.inputs}
        if reachable_ids - declared_ids:
            missing = [n.op[1] for n in reachable if id(n) not in declared_ids]
            raise ValueError(f"outputs reference undeclared inputs: {missing}")
        if declared_ids - reachable_ids:
            unused = [n.op[1] for n in self.inputs if id(n) not in reachable_ids]
            raise ValueError(f"declared inputs are unused by any output: {unused}")
        _check_stateful_safety(self.outputs)
        self._names = [n.op[1] for n in self.inputs]
        self._cg, self._input_order = self._compile_cpp()

    def _compile_cpp(self):
        from . import screamer_bindings as _b
        gb = _b._GraphBuilder()
        ids = {}   # id(node) -> C++ node id

        def build(node):
            node_id = id(node)
            if node_id in ids:
                return ids[node_id]
            op = node.op
            if isinstance(op, tuple) and op[0] == "input":
                nid = gb.add_input()
            elif isinstance(op, tuple) and op[0] == "operator":
                fn, kwargs = op[1], op[2]
                name = getattr(fn, "__name__", "")
                inp = [build(i) for i in node.inputs]
                if name == "CombineLatest":
                    nid = gb.add_combine_latest(inp, kwargs.get("emit") == "when_all")
                elif name == "Filter":
                    nid = gb.add_filter(inp)
                elif name == "Dropna":
                    nid = gb.add_dropna(inp, kwargs.get("how") == "all")
                elif name == "Select":
                    from .streams import _normalize_columns
                    cols, _ = _normalize_columns(kwargs["columns"])
                    nid = gb.add_select(inp, cols)
                elif name == "Resample":
                    mode = 1 if kwargs.get("count") is not None else 0   # 0=ByIndex,1=ByCount
                    label = 1 if kwargs.get("label", "left") == "right" else 0
                    width = int(kwargs["every"]) if kwargs.get("every") is not None else 1
                    origin = int(kwargs.get("origin", 0))
                    count = int(kwargs["count"]) if kwargs.get("count") is not None else 1
                    agg_val = kwargs.get("agg", "last")
                    fill = _RESAMPLE_FILL_CODE[kwargs.get("fill", "skip")]
                    if isinstance(agg_val, str):
                        # Builtin string agg: enum code, no functor reducer.
                        nid = gb.add_resample(inp, mode, _RESAMPLE_AGG_CODE[agg_val],
                                              label, width, origin, count, fill=fill)
                    else:
                        # Arbitrary functor reducer (an EvalOp): agg code is ignored
                        # (pass 0); the reducer op drives GenericResampleNode.
                        nid = gb.add_resample(inp, mode, 0, label, width, origin,
                                              count, agg_val, fill=fill)
                else:
                    raise ValueError(
                        f"{name} is not supported as a DAG graph node")
            else:                                    # functor instance (EvalOp)
                inp = [build(i) for i in node.inputs]
                nid = gb.add_functor(op, inp)
            ids[node_id] = nid
            return nid

        # build Input nodes first so their ids follow signature order
        for n in self.inputs:
            build(n)
        out_ids = [build(o) for o in self.outputs]
        gb.set_outputs(out_ids)
        # map signature order -> the add_input order (they match: inputs built first)
        return gb.compile(), list(self._names)

    def __call__(self, *args, **kwargs):
        feeds = self._bind_args(args, kwargs)
        lazy = [self._is_lazy(v) for v in feeds.values()]
        if feeds and all(lazy):
            return _LazyDag(self, feeds)
        if any(lazy):
            raise TypeError(
                "dag(...) feeds must be either all lazy iterators or all concrete "
                "(arrays / lists / (values, index) tuples); a mix is ambiguous. "
                "Wrap the concrete feed in a generator, or materialize the "
                "iterator into an array.")
        streams = [_as_stream(feeds[nm]) for nm in self._input_order]
        results = self._cg.run_batch(streams)      # M independent (index, values2d)
        results = [(k, v.reshape(-1) if v.shape[1] == 1 else v) for (k, v) in results]
        return _align_results(results, self.align_outputs)

    @staticmethod
    def _is_lazy(x):
        # A feed is lazy iff it is an iterator (has __next__) and is not a
        # list, tuple, or ndarray (those are concrete/batch, per rule A).
        return hasattr(x, "__next__") and not isinstance(x, (list, tuple, np.ndarray))

    def live(self):
        """Open a live streaming session: push events and drive a clock yourself.

        Bind data AFTER definition, event by event, on the same Pipeline that runs batch.
        Push events with .push(input, index, value); close windows whose boundary has
        passed with .advance(now) (e.g. on a clock tick, finalizing empty bars); force
        the current partial window with .flush(); collect aligned outputs with .result().

        The session drives this Pipeline's single compiled engine (shared with __call__),
        resetting it on open, so use one session at a time: do not interleave a live
        session with a batch call or run two live sessions concurrently.
        """
        return _LiveDag(self)

    # -- inspection / visualization ------------------------------------------
    def __repr__(self):
        return f"Pipeline({len(self.inputs)} input(s), {len(self.outputs)} output(s))"

    def __str__(self):
        from .dag_viz import to_text
        return to_text(self)

    def to_text(self):
        """An indented, ``tree``-style text view of the graph (zero dependencies)."""
        from .dag_viz import to_text
        return to_text(self)

    def to_dot(self):
        """A Graphviz DOT string for the graph (zero dependencies)."""
        from .dag_viz import to_dot
        return to_dot(self)

    def to_graphviz(self):
        """A ``graphviz.Source`` for the graph (needs the optional ``graphviz`` package)."""
        from .dag_viz import to_graphviz
        return to_graphviz(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        from .dag_viz import repr_mimebundle
        return repr_mimebundle(self)

    # -- serialization -------------------------------------------------------
    def to_dict(self):
        """A JSON-native dict describing the graph (round-trips via :meth:`from_dict`)."""
        from .dag_io import to_dict
        return to_dict(self)

    def to_json(self, indent=2):
        """A JSON string describing the graph (round-trips via :meth:`from_json`)."""
        from .dag_io import to_json
        return to_json(self, indent)

    @classmethod
    def from_dict(cls, data):
        """Rebuild a runnable ``Pipeline`` from a dict produced by :meth:`to_dict`."""
        from .dag_io import from_dict
        return from_dict(data)

    @classmethod
    def from_json(cls, text):
        """Rebuild a runnable ``Pipeline`` from JSON produced by :meth:`to_json`."""
        from .dag_io import from_json
        return from_json(text)

    def _bind_args(self, args, kwargs):
        if args and kwargs:
            raise TypeError("pass inputs either positionally or by name, not both")
        if kwargs:
            missing = [nm for nm in self._names if nm not in kwargs]
            extra = [k for k in kwargs if k not in self._names]
            if missing or extra:
                raise TypeError(f"input mismatch: missing={missing} unknown={extra}")
            return {nm: kwargs[nm] for nm in self._names}
        if len(args) != len(self._names):
            raise TypeError(
                f"expected {len(self._names)} inputs, got {len(args)}")
        return {nm: val for nm, val in zip(self._names, args)}
