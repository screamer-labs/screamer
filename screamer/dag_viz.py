"""Render a ``Dag`` for inspection: an ASCII tree, Graphviz DOT, and a Jupyter
repr. The graph walk here (``build_graph``) is also reused by ``dag_io`` for
serialization, so labels and structure stay consistent.
"""
from collections import namedtuple

from ._functor_params import format_call

# One node of the graph, in dependency order (inputs before dependents).
#   id      : stable integer id
#   kind    : "input" | "operator" | "functor"
#   name    : input name / operator function name / functor class name
#   params  : dict of parameters (operator kwargs or functor _screamer_params)
#   in_ids  : ids of the nodes feeding this one
NodeInfo = namedtuple("NodeInfo", "id kind name params in_ids")


def _classify(node):
    """Return (kind, name, params) for a graph Node."""
    op = node.op
    if isinstance(op, tuple) and op and op[0] == "input":
        return "input", op[1], {}
    if isinstance(op, tuple) and op and op[0] == "operator":
        fn, kwargs = op[1], op[2]
        return "operator", getattr(fn, "__name__", "operator"), dict(kwargs)
    return "functor", type(op).__name__, dict(getattr(op, "_screamer_params", {}))


def build_graph(dag):
    """Walk the Dag's outputs into an ordered node list plus the input/output ids.

    Nodes are deduplicated by identity (a shared node appears once) and numbered
    in dependency order. Returns (nodes, input_ids, output_ids).
    """
    order, id_of = [], {}

    def visit(node):
        key = id(node)
        if key in id_of:
            return id_of[key]
        in_ids = [visit(i) for i in node.inputs]
        kind, name, params = _classify(node)
        nid = len(order)
        id_of[key] = nid
        order.append(NodeInfo(nid, kind, name, params, in_ids))
        return nid

    for out in dag.outputs:
        visit(out)
    output_ids = [id_of[id(o)] for o in dag.outputs]
    input_ids = [id_of[id(n)] for n in dag.inputs]
    return order, input_ids, output_ids


def _label(node):
    if node.kind == "input":
        return str(node.name)
    # Drop None-valued params (internal placeholders) from the display label.
    params = {k: v for k, v in node.params.items() if v is not None}
    return format_call(node.name, params)


# --------------------------------------------------------------------------- #
# ASCII tree
# --------------------------------------------------------------------------- #

def to_text(dag):
    """An indented, tree-style view rooted at each output, descending to inputs.

    A shared node that has inputs of its own (an operator or functor) is expanded
    once and later referenced by id (``^#3``), so a diamond reads as a diamond
    instead of duplicating a whole subtree. Leaf inputs, having no subtree, are
    just reprinted where they feed.
    """
    nodes, input_ids, output_ids = build_graph(dag)
    info = {n.id: n for n in nodes}
    header = (f"Dag({len(input_ids)} input(s), {len(output_ids)} output(s), "
              f"align_outputs={dag.align_outputs})")
    lines = [header]
    printed = set()

    def render(nid, prefix, is_last, root_label=None):
        node = info[nid]
        connector = "" if root_label is not None else ("└─ " if is_last else "├─ ")
        head = f"{root_label} = " if root_label is not None else ""
        if nid in printed and node.in_ids:
            lines.append(f"{prefix}{connector}{head}^#{nid} {node.name}")
            return
        printed.add(nid)
        tag = f"  #{nid}" if node.kind != "input" else f"  #{nid} (input)"
        lines.append(f"{prefix}{connector}{head}{_label(node)}{tag}")
        child_prefix = prefix + ("   " if (is_last or root_label is not None) else "│  ")
        for i, cid in enumerate(node.in_ids):
            render(cid, child_prefix, i == len(node.in_ids) - 1)

    for j, oid in enumerate(output_ids):
        render(oid, "", True, root_label=f"out[{j}]")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Graphviz
# --------------------------------------------------------------------------- #

_INPUT_STYLE = 'shape=ellipse, style=filled, fillcolor="#e8f0fe"'
_OPERATOR_STYLE = 'shape=box, style="rounded,filled", fillcolor="#fff3e0"'
_FUNCTOR_STYLE = 'shape=box, style=filled, fillcolor="#f5f5f5"'


def to_dot(dag):
    """A Graphviz DOT string (no dependency required to produce it)."""
    nodes, _input_ids, output_ids = build_graph(dag)
    output_set = set(output_ids)
    lines = ["digraph screamer_dag {", "  rankdir=LR;",
             '  node [fontname="monospace"];']
    for n in nodes:
        label = _label(n).replace("\\", "\\\\").replace('"', '\\"')
        style = {"input": _INPUT_STYLE, "operator": _OPERATOR_STYLE}.get(
            n.kind, _FUNCTOR_STYLE)
        if n.id in output_set:
            style += ', penwidth=2, color="#1a73e8"'
        lines.append(f'  n{n.id} [label="{label}", {style}];')
    for n in nodes:
        for cid in n.in_ids:
            lines.append(f"  n{cid} -> n{n.id};")
    lines.append("}")
    return "\n".join(lines)


def to_graphviz(dag):
    """A ``graphviz.Source`` for the DAG (renders to SVG/PNG, shows inline in Jupyter).

    Requires the optional ``graphviz`` package (and the system ``dot`` binary to
    render). ``to_dot()`` works without it.
    """
    try:
        import graphviz
    except ImportError as exc:
        raise ImportError(
            "to_graphviz() needs the 'graphviz' package: pip install screamer[viz] "
            "(and the system Graphviz 'dot' binary). to_dot() works without it."
        ) from exc
    return graphviz.Source(to_dot(dag))


def repr_mimebundle(dag):
    """Jupyter rich display: inline SVG when graphviz renders, else the ASCII text."""
    bundle = {"text/plain": to_text(dag)}
    try:
        import graphviz
        svg = graphviz.Source(to_dot(dag)).pipe(format="svg").decode("utf-8")
        bundle["image/svg+xml"] = svg
    except Exception:
        pass
    return bundle
