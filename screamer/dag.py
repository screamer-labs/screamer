"""Computational DAG definition: symbolic Node handles (DAG-1)."""

import numpy as np

__all__ = ["Node", "Input", "Pipeline"]

_RESAMPLE_AGG_CODE = {"first": 0, "last": 1, "min": 2, "max": 3,
                      "sum": 4, "count": 5, "mean": 6, "ohlc": 7,
                      "ohlcv_bars": 10}   # 10 = OhlcvBars (dynamic plan, built in C++)
# ResampleMode enum codes (matching C++ enum ResampleMode).
_RESAMPLE_MODE_CODE = {"by_index": 0, "by_count": 1, "by_cumulative": 2, "by_clock": 3}
_RESAMPLE_FILL_CODE = {"skip": 0, "nan": 1, "carry": 2}

# ResampleAgg enum codes for plan entries (matching the C++ enum).
_PLAN_AGG = {
    "first": 0, "last": 1, "min": 2, "max": 3,
    "sum": 4, "sum_pos": 8, "sum_neg": 9,
}

# Per-column reducer plans for multi-column bar aggs. Each entry is a list of
# (agg_code, input_col_index) tuples that map output columns to reducers.
# ohlc_bars: [first O, max H, min L, last C] - 4 in, 4 out
# ohlcv_bars: plan is dynamic (N in, N out); C++ builds it from the input width.
# ohlcv:  [first price, max price, min price, last price, sum vol] - 2 in, 5 out
# ohlcv2: [first price, max price, min price, last price, sumPos vol, sumNeg vol]
_BAR_AGG_FIXED_PLANS = {
    "ohlc_bars": [(0, 0), (3, 1), (2, 2), (1, 3)],   # first O, max H, min L, last C
    "ohlcv":     [(0, 0), (3, 0), (2, 0), (1, 0), (4, 1)],   # OHLC from col0, sum from col1
    "ohlcv2":    [(0, 0), (3, 0), (2, 0), (1, 0), (8, 1), (9, 1)],  # OHLC from col0, pos/neg from col1
}
# ohlcv_bars uses agg code 10 (OhlcvBars) with an empty plan; C++ builds the plan
# dynamically from the actual input width at ResampleNode instantiation time.


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


def _index_array(idx):
    """An index as a contiguous array for the engine, which applies the index rule.

    The C++ engine checks every element in its int64 cast loop: an index must be
    finite, whole and inside the int64 range, and an array of bool, str, object or
    complex values is a TypeError. A datetime64 or timedelta64 index is read as
    its int64 ticks, and NaT raises TypeError.
    """
    arr = np.ascontiguousarray(idx)
    if not arr.dtype.isnative:
        arr = arr.astype(arr.dtype.newbyteorder("="))
    return arr


def _float_values(values):
    """A feed's values as contiguous float64; complex values raise TypeError, as
    operators do (the float64 cast would drop the imaginary part)."""
    # A value with a dtype (an array, a pandas Series) is cast as it is, so pandas
    # maps NA to NaN; one without (a list) is read by NumPy first to find its kind.
    if isinstance(values, (list, tuple)):
        _refuse_text(values)       # before NumPy reads a bytearray item as numbers
    arr = values if hasattr(values, "dtype") else np.asarray(values)
    if getattr(arr.dtype, "kind", None) == "c":
        raise TypeError("a value must be a real number or a sequence of real "
                        "numbers, got complex values")
    return _as_float64(arr, contiguous=True)


def _refuse_text(x):
    """The C++ string rule (cast_double.h refuse_text): a string, or a list or
    array holding one, raises TypeError."""
    from . import screamer_bindings as _b
    _b._refuse_text(x)


def _as_float64(x, contiguous=False):
    """`x` as a float64 array (np.asarray, or np.ascontiguousarray when
    `contiguous`). A string, or an array or list holding one, raises TypeError
    (a string is never a number); None is NaN. The string check runs in C++ and
    only for a string, object or unknown kind, so a numeric array pays nothing."""
    kind = getattr(getattr(x, "dtype", None), "kind", None)
    if kind is None and not isinstance(x, (float, int)):
        if isinstance(x, (list, tuple)):
            _refuse_text(x)        # before NumPy reads a bytearray item as numbers
        x = np.asarray(x)
        kind = x.dtype.kind
    if kind in ("U", "S", "O"):
        _refuse_text(x)
    return (np.ascontiguousarray if contiguous else np.asarray)(x, dtype=np.float64)


def _setting(name, x):
    """An operator setting that must be a number: a string is a wrong kind
    (TypeError), never parsed. Returns `x`."""
    if isinstance(x, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a number, got {type(x).__name__}")
    return x


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
        values_arr = _float_values(values)
        if index is None:
            return (np.arange(values_arr.shape[0], dtype=np.int64), values_arr)
        return (_index_array(index), values_arr)
    values = _float_values(feed)
    return (np.arange(values.shape[0], dtype=np.int64), values)


def _values_first(results):
    """The engine's (index, values2d) pairs as (values, index) pairs; a
    one-column output is 1-D. One pair for one output, else a tuple of pairs."""
    out = tuple((v.reshape(-1) if v.shape[1] == 1 else v, k) for k, v in results)
    return out[0] if len(out) == 1 else out


def _aligned(result):
    """The engine's aligned outputs (index, [values2d, ...]) as (values, index)
    pairs that share the one index; a one-column output is 1-D."""
    index, values = result
    return tuple((v.reshape(-1) if v.shape[1] == 1 else v, index) for v in values)


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


_LIVE_CLASS = None


def _live_dag_class():
    """The live-session class, built on first use (the bindings load lazily)."""
    global _LIVE_CLASS
    if _LIVE_CLASS is not None:
        return _LIVE_CLASS
    from . import screamer_bindings as _b

    class _LiveDag(_b._LiveSession):
        """A live streaming session over a compiled Pipeline. `input` in push() is an Input
        name (str) or its positional index (int). push() is a C++ method: `value` is a
        number, or a sequence of numbers for one multi-column event."""
        def __init__(self, dag):
            super().__init__(dag._cg, list(dag._input_order))
            self._dag = dag
            self._cg = dag._cg
            self._cg.reset()

        def advance(self, now):
            """Close every window whose boundary has passed by logical time `now`."""
            self._advance(now)
            return self

        def flush(self):
            """Finalize the current partial window(s) now (end-of-loop / on demand)."""
            self._cg.flush()
            return self

        def result(self):
            """Aligned outputs accumulated so far (drains the buffers). Raises after a
            rejected event poisoned the session, until a new live() session."""
            dag = self._dag
            if len(dag.outputs) > 1 and dag.align_outputs:
                return _aligned(self._cg.drain(True))
            return _values_first(self._cg.drain())

    _LIVE_CLASS = _LiveDag
    return _LIVE_CLASS


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
    - ``max_pending`` (default ``1_000_000``): the most events a join may hold
      while it waits for another stream (each ``CombineLatest`` and ``Filter``
      node, and the aligned outputs of a lazy call or of ``live().result()``).
      More raises ``RuntimeError``. An integer from 1 to ``2**53``.

    Call ``pipe(*feeds)`` (positional) or ``pipe(**named_feeds)`` (by Input name)
    to run the pipeline. Each feed may be a bare value array (positional, index
    = row-number), or a ``(values, index)`` pair (values-first).
    Pass generators of ``(value, index)`` pairs to run the pipeline lazily, event
    by event, with byte-identical results (the lazy pull path).
    Returns a single ``(values, index)`` pair when M == 1, or a tuple of pairs
    when M > 1.
    """

    def __init__(self, inputs, outputs, align_outputs=True, max_pending=1_000_000):
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.align_outputs = align_outputs
        from . import screamer_bindings as _b
        self.max_pending = _b._max_pending(max_pending)
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
        gb.set_max_pending(self.max_pending)
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
                    nid = gb.add_combine_latest(inp, kwargs.get("emit") == "when_all",
                                               self.max_pending)
                elif name == "Filter":
                    nid = gb.add_filter(inp)
                elif name == "Dropna":
                    nid = gb.add_dropna(inp, kwargs.get("how") == "all")
                elif name == "Select":
                    from .streams import _normalize_columns
                    cols, _ = _normalize_columns(kwargs["columns"])
                    nid = gb.add_select(inp, cols)
                elif name == "Resample":
                    for key in ("threshold", "every", "count", "origin", "max_age"):
                        _setting(key, kwargs.get(key))
                    threshold = kwargs.get("threshold")
                    max_age = kwargs.get("max_age")
                    clock = kwargs.get("clock", False)
                    if clock:
                        mode = 3  # ByClock
                    elif threshold is not None:
                        mode = 2  # ByCumulative
                    elif kwargs.get("count") is not None:
                        mode = 1  # ByCount
                    else:
                        mode = 0  # ByIndex
                    label = 1 if kwargs.get("label", "left") == "right" else 0
                    width = int(kwargs["every"]) if kwargs.get("every") is not None else 1
                    origin = int(kwargs.get("origin", 0))
                    count = int(kwargs["count"]) if kwargs.get("count") is not None else 1
                    agg_val = kwargs.get("agg", "last")
                    fill = _RESAMPLE_FILL_CODE[kwargs.get("fill", "skip")]
                    threshold_val = float(threshold) if threshold is not None else 0.0
                    max_age_val = int(max_age) if max_age is not None else -1
                    if isinstance(agg_val, str) and agg_val in _BAR_AGG_FIXED_PLANS:
                        # Multi-column bar agg with a fixed plan (ohlc_bars, ohlcv, ohlcv2).
                        plan = _BAR_AGG_FIXED_PLANS[agg_val]
                        nid = gb.add_resample(inp, mode, 0, label, width, origin,
                                              count, fill=fill, plan=plan,
                                              threshold=threshold_val,
                                              max_age=max_age_val,
                                              max_pending=self.max_pending)
                    elif isinstance(agg_val, str) and agg_val == "ohlcv_bars":
                        # ohlcv_bars: plan depends on input width (unknown at graph-build
                        # time). Pass agg=10 (OhlcvBars enum value) with an empty plan;
                        # the C++ compiled graph builds make_ohlcv_bars_plan(input_width)
                        # at ResampleNode instantiation from the resolved input width.
                        nid = gb.add_resample(inp, mode, 10, label, width, origin,
                                              count, fill=fill, threshold=threshold_val,
                                              max_age=max_age_val,
                                              max_pending=self.max_pending)
                    elif isinstance(agg_val, str):
                        # Builtin string agg: enum code, no functor reducer.
                        nid = gb.add_resample(inp, mode, _RESAMPLE_AGG_CODE[agg_val],
                                              label, width, origin, count, fill=fill,
                                              threshold=threshold_val,
                                              max_age=max_age_val,
                                              max_pending=self.max_pending)
                    else:
                        # Arbitrary functor reducer (an EvalOp): agg code is ignored
                        # (pass 0); the reducer op drives GenericResampleNode.
                        nid = gb.add_resample(inp, mode, 0, label, width, origin,
                                              count, agg_val, fill=fill,
                                              threshold=threshold_val,
                                              max_age=max_age_val,
                                              max_pending=self.max_pending)
                elif name == "Delay":
                    nid = gb.add_delay(inp, int(_setting("duration", kwargs["duration"])))
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
            # The C++ driver merges the feeds, drives the graph, and yields rows:
            # (value, index) for one output, or (col0, ..., col_{M-1}, index) for
            # M>1 aligned outputs (the engine's output alignment, as in batch).
            # Independent (align_outputs=False) multi-output has no single lazy
            # row shape.
            if len(self.outputs) > 1 and not self.align_outputs:
                raise NotImplementedError(
                    "lazy dag(iterables) with align_outputs=False and multiple "
                    "outputs is not yet supported; use the batch call dag(arrays)")
            from . import screamer_bindings as _b
            iters = [iter(feeds[nm]) for nm in self._input_order]
            return _b._LazyDriver(self._cg, iters, len(self.outputs))
        if any(lazy):
            raise TypeError(
                "dag(...) feeds must be either all lazy iterators or all concrete "
                "(arrays / lists / (values, index) tuples); a mix is ambiguous. "
                "Wrap the concrete feed in a generator, or materialize the "
                "iterator into an array.")
        streams = [_as_stream(feeds[nm]) for nm in self._input_order]
        if len(self.outputs) > 1 and self.align_outputs:
            return _aligned(self._cg.run_batch(streams, True))
        return _values_first(self._cg.run_batch(streams))

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
        return _live_dag_class()(self)

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
