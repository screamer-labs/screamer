"""Computational DAG definition: symbolic Node handles (DAG-1)."""

import numpy as np

__all__ = ["Node", "Input", "Dag"]


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
        True  – co-index all M outputs onto a shared, sorted key axis.
                 Each output carries its as-of value at every unique union key
                 (combine_latest's per-event intermediate rows are collapsed to
                 one row per key).  Returns a tuple of (keys, values) pairs of
                 equal length.
        False – return independent per-output streams; lengths may differ.

    Calling
    -------
    dag(*feeds)  or  dag(**named_feeds)
        Evaluate the graph.  Returns a single (keys, values) pair when M == 1,
        or a tuple of (keys, values) pairs when M > 1.
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

    def __call__(self, *args, **kwargs):
        feeds = self._bind_args(args, kwargs)
        return self._run(feeds)          # implemented in Task 5

    def _run(self, feeds):
        memo = {}

        def ev(node):
            key = id(node)
            if key in memo:
                return memo[key]
            op = node.op
            if isinstance(op, tuple) and op[0] == "input":
                result = _as_stream(feeds[op[1]])
            elif isinstance(op, tuple) and op[0] == "combinator":
                fn, kwargs = op[1], op[2]
                result = fn(*[ev(i) for i in node.inputs], **kwargs)
            else:                                   # functor instance
                ins = [ev(i) for i in node.inputs]
                out_keys = ins[0][0]
                out_vals = op(*[v for (_, v) in ins])
                result = (out_keys, out_vals)
            memo[key] = result
            return result

        results = [ev(o) for o in self.outputs]
        if len(results) == 1:
            return results[0]
        if not self.align_outputs:
            return tuple(results)
        # align_outputs: combine_latest the M outputs onto a shared key axis,
        # then deduplicate to one row per unique key (last/most up-to-date value).
        # combine_latest is imported lazily to avoid the streams→dag import cycle.
        from .streams import combine_latest
        aligned_keys, aligned = combine_latest(*results, emit="when_all")
        # Keep only the last row for each unique key so that all outputs carry
        # their final value at that key (forward-fill artefacts are dropped).
        # combine_latest emits in non-decreasing key order, so index order == key order;
        # sorting last-occurrence indices preserves key order.
        _, inv_idx = np.unique(aligned_keys[::-1], return_index=True)
        last_idx = np.sort(len(aligned_keys) - 1 - inv_idx)
        aligned_keys = aligned_keys[last_idx]
        aligned = aligned[last_idx]
        return tuple((aligned_keys, aligned[:, j]) for j in range(len(results)))

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


