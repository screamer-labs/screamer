"""Computational DAG definition: symbolic Node handles (DAG-1)."""

import numpy as np

__all__ = ["Node", "Input", "Dag"]

_RESAMPLE_AGG_CODE = {"first": 0, "last": 1, "min": 2, "max": 3,
                      "sum": 4, "count": 5, "mean": 6, "ohlc": 7}


class Node:
    """An immutable handle for a stream in a computation graph.

    op is one of:
      - ("input", name)
      - a configured functor instance (ScreamerBase / FunctorBase)
      - ("combinator", fn, kwargs)
    inputs is a tuple of upstream Nodes.
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
        if isinstance(self.op, tuple) and self.op and self.op[0] == "combinator":
            name = self.op[1].__name__
        else:
            name = type(self.op).__name__
        return f"Node({name}, {len(self.inputs)} input(s))"


def is_node(obj):
    return getattr(obj, "is_node", False) is True


def Input(name):
    """Create a source Node — a named placeholder for a timed stream."""
    return Node(("input", name))


def make_functor_node(functor, args):
    """Build a Node for a functor applied to Node args. Called by the C++ hook."""
    inputs = tuple(args)
    for n in inputs:
        if not is_node(n):
            raise TypeError("all arguments must be Nodes when building a graph")
    return Node(functor, inputs)


def make_combinator_node(fn, node_args, kwargs):
    """Build a Node for a combinator applied to Node args."""
    return Node(("combinator", fn, kwargs), tuple(node_args))


def _as_stream(feed):
    """Normalize a feed to a (keys, values) stream tuple.

    Accepts a bare value array (keys = row-number int64) or a
    (keys, values) pair and returns the canonical (keys, values) form.
    """
    if isinstance(feed, tuple) and len(feed) == 2:
        return feed
    values = np.asarray(feed, dtype=np.float64)
    return (np.arange(values.shape[0], dtype=np.int64), values)


def _align_results(results, align_outputs):
    """Boundary align: single stream for M==1; tuple of independent streams for
    align_outputs=False; co-indexed tuple (combine_latest + one-row-per-key) for
    align_outputs=True. Operates on already-computed (keys, values) output streams.
    """
    if len(results) == 1:
        return results[0]
    if not align_outputs:
        return tuple(results)
    from .streams import combine_latest
    aligned_keys, aligned = combine_latest(*results, emit="when_all")
    _, inv_idx = np.unique(aligned_keys[::-1], return_index=True)
    last_idx = np.sort(len(aligned_keys) - 1 - inv_idx)
    aligned_keys = aligned_keys[last_idx]
    aligned = aligned[last_idx]
    return tuple((aligned_keys, aligned[:, j]) for j in range(len(results)))


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


class Dag:
    """A positional N-in / M-out callable that evaluates a computation graph.

    Parameters
    ----------
    inputs : list[Node]
        Ordered list of Input(...) nodes that define the call signature.
        Feeds are bound positionally (or by name via keyword call).
    outputs : list[Node]
        Ordered list of output nodes to evaluate.
    align_outputs : bool, default True
        When True (default), co-index all M outputs onto a shared, sorted key
        axis: each output carries its as-of value at every unique union key
        (combine_latest's per-event intermediate rows are collapsed to one row
        per key), returning a tuple of equal-length (keys, values) pairs. When
        False, return independent per-output streams whose lengths may differ.

    Notes
    -----
    Call ``dag(*feeds)`` (positional) or ``dag(**named_feeds)`` (by Input name)
    to evaluate the graph. Returns a single (keys, values) pair when M == 1, or a
    tuple of (keys, values) pairs when M > 1. Use ``dag.stream(*feeds)`` to run
    the same graph live, event by event, with byte-identical results.
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
            key = id(node)
            if key in ids:
                return ids[key]
            op = node.op
            if isinstance(op, tuple) and op[0] == "input":
                nid = gb.add_input()
            elif isinstance(op, tuple) and op[0] == "combinator":
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
                    mode = 1 if kwargs.get("count") is not None else 0   # 0=ByKey,1=ByCount
                    agg = _RESAMPLE_AGG_CODE[kwargs.get("agg", "last")]
                    label = 1 if kwargs.get("label", "left") == "right" else 0
                    width = int(kwargs["width"]) if kwargs.get("width") is not None else 1
                    origin = int(kwargs.get("origin", 0))
                    count = int(kwargs["count"]) if kwargs.get("count") is not None else 1
                    nid = gb.add_resample(inp, mode, agg, label, width, origin, count)
                else:
                    raise ValueError(
                        f"{name} is not supported as a DAG graph node")
            else:                                    # functor instance (EvalOp)
                inp = [build(i) for i in node.inputs]
                nid = gb.add_functor(op, inp)
            ids[key] = nid
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
        results = self._cg.run_batch(streams)      # M independent (keys, values2d)
        results = [(k, v.reshape(-1) if v.shape[1] == 1 else v) for (k, v) in results]
        return _align_results(results, self.align_outputs)

    def stream(self, *args, **kwargs):
        """Drive the compiled graph live, event by event (byte-identical to __call__)."""
        from .streams import merge
        feeds = self._bind_args(args, kwargs)
        streams = [_as_stream(feeds[nm]) for nm in self._input_order]
        self._cg.reset()
        # feed the merged (key-ordered, source-tagged) events one at a time
        mk, mv, ms = merge(*streams)
        for k, v, s in zip(mk, mv, ms):
            self._cg.push_event(int(s), int(k), float(v))
        self._cg.flush()          # end-of-input: emit trailing resample buckets
        results = self._cg.drain()
        results = [(k, v.reshape(-1) if v.shape[1] == 1 else v) for (k, v) in results]
        return _align_results(results, self.align_outputs)

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


