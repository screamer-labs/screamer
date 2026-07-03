"""Computational DAG definition: symbolic Node handles (DAG-1)."""

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


