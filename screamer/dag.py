"""Computational DAG definition: symbolic Node handles (DAG-1)."""

import numpy as np

__all__ = ["Node", "Input", "Dag"]

_RESAMPLE_AGG_CODE = {"first": 0, "last": 1, "min": 2, "max": 3,
                      "sum": 4, "count": 5, "mean": 6, "ohlc": 7}
_RESAMPLE_FILL_CODE = {"skip": 0, "nan": 1, "carry": 2}


class Node:
    """An immutable handle for a stream in a computation graph.

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


def _as_stream(feed):
    """Normalize a feed to (index_array, values_array) for the compiled engine.

    Accepts:
    - bare value array -> positional (index = row-number int64 via np.arange)
    - Stream          -> uses .values / .index (.index None -> row-number)
    - (values, index) pair -> values-first user convention; flipped for engine
    """
    from .streams import Stream
    if isinstance(feed, Stream):
        idx = feed.index
        if idx is None:
            idx = np.arange(len(feed.values), dtype=np.int64)
        return (
            np.ascontiguousarray(idx, dtype=np.int64),
            np.ascontiguousarray(feed.values, dtype=np.float64),
        )
    if isinstance(feed, tuple) and len(feed) == 2:
        values, index = feed   # user provides (values, index) - values-first
        return (
            np.ascontiguousarray(index, dtype=np.int64),
            np.ascontiguousarray(values, dtype=np.float64),
        )
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
    from .streams import combine_latest, Stream
    # wrap in Streams; combine_latest coalesces (one row per distinct index)
    stream_list = [Stream(v, k) for k, v in results]
    out = combine_latest(*stream_list, emit="when_all")   # returns Stream
    aligned_index = out.index   # one row per distinct index - already coalesced
    aligned = out.values        # shape (N, M)
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
    """A live streaming session over a compiled Dag. `input` in push() is an Input
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


class Dag:
    """A positional N-in / M-out callable that evaluates a computation graph.

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

    Call ``dag(*feeds)`` (positional) or ``dag(**named_feeds)`` (by Input name)
    to evaluate the graph. Each feed may be a bare value array (positional, index
    = row-number), a ``Stream``, or a ``(values, index)`` pair (values-first).
    Returns a single ``(values, index)`` pair when M == 1, or a tuple of pairs
    when M > 1. Use ``dag.stream(*feeds)`` to run the same graph live, event by
    event, with byte-identical results.
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
                if name == "combine_latest":
                    nid = gb.add_combine_latest(inp, kwargs.get("emit") == "when_all")
                elif name == "dropna":
                    nid = gb.add_dropna(inp, kwargs.get("how") == "all")
                elif name == "select":
                    from .streams import _normalize_columns
                    cols, _ = _normalize_columns(kwargs["columns"])
                    nid = gb.add_select(inp, cols)
                elif name == "resample":
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
                elif name == "multi_resample":
                    mode = 1 if kwargs.get("count") is not None else 0
                    label = 1 if kwargs.get("label", "left") == "right" else 0
                    width = int(kwargs["every"]) if kwargs.get("every") is not None else 1
                    origin = int(kwargs.get("origin", 0))
                    count = int(kwargs["count"]) if kwargs.get("count") is not None else 1
                    fill = _RESAMPLE_FILL_CODE[kwargs.get("fill", "skip")]
                    nid = gb.add_multicolumn_resample(inp, mode, label, width, origin,
                                                      count, fill, kwargs["reducers"])
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
        streams = [_as_stream(feeds[nm]) for nm in self._input_order]
        results = self._cg.run_batch(streams)      # M independent (index, values2d)
        results = [(k, v.reshape(-1) if v.shape[1] == 1 else v) for (k, v) in results]
        return _align_results(results, self.align_outputs)

    def stream(self, *args, **kwargs):
        """Drive the compiled graph live, event by event (byte-identical to __call__)."""
        from .streams import merge
        feeds = self._bind_args(args, kwargs)
        streams = [_as_stream(feeds[nm]) for nm in self._input_order]
        self._cg.reset()
        # split (index_arr, values_arr) pairs so merge can align them by index
        idx_arrays = [s[0] for s in streams]
        val_arrays = [s[1] for s in streams]
        merged_vals, merged_sources, merged_index = merge(*val_arrays, index=idx_arrays)
        for v, src, k in zip(merged_vals, merged_sources, merged_index):
            self._cg.push_event(int(src), int(k), float(v))
        self._cg.flush()          # end-of-input: emit trailing resample buckets
        results = self._cg.drain()
        results = [(k, v.reshape(-1) if v.shape[1] == 1 else v) for (k, v) in results]
        return _align_results(results, self.align_outputs)

    def live(self):
        """Open a live streaming session: push events and drive a clock yourself.

        Bind data AFTER definition, event by event - the same Dag that runs batch.
        Push events with .push(input, index, value); close windows whose boundary has
        passed with .advance(now) (e.g. on a clock tick, finalizing empty bars); force
        the current partial window with .flush(); collect aligned outputs with .result().
        """
        return _LiveDag(self)

    # -- inspection / visualization ------------------------------------------
    def __repr__(self):
        return f"Dag({len(self.inputs)} input(s), {len(self.outputs)} output(s))"

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
        """Rebuild a runnable ``Dag`` from a dict produced by :meth:`to_dict`."""
        from .dag_io import from_dict
        return from_dict(data)

    @classmethod
    def from_json(cls, text):
        """Rebuild a runnable ``Dag`` from JSON produced by :meth:`to_json`."""
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
