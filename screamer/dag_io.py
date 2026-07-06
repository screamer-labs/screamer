"""Serialize a ``Dag`` to a JSON-native dict (and back).

A Dag is captured as its ordered input names, its nodes (each input / operator /
functor with params and the ids of the nodes feeding it), its outputs, and
``align_outputs``. ``from_dict`` rebuilds a runnable Dag, so a graph round-trips
as a config file. Reuses the same graph walk as the visualizer so structure and
labels stay consistent.
"""
import json

from . import streams
from .dag import Dag, Input
from .dag_viz import build_graph

SCHEMA_VERSION = 1

# The four operators that can be graph nodes (mirrors Dag._compile_cpp's support).
_OPERATORS = {
    "combine_latest": streams.combine_latest,
    "dropna": streams.dropna,
    "select": streams.select,
    "resample": streams.resample,
}


def _clean(params):
    """Drop None-valued params (defaults / internal placeholders) for a compact config."""
    return {k: v for k, v in params.items() if v is not None}


def to_dict(dag):
    """Return a JSON-native dict fully describing the graph."""
    nodes, input_ids, output_ids = build_graph(dag)
    id_to_name = {n.id: n.name for n in nodes if n.kind == "input"}
    serialized = []
    for n in nodes:
        if n.kind == "input":
            serialized.append({"id": n.id, "kind": "input", "name": n.name})
        elif n.kind == "operator":
            serialized.append({"id": n.id, "kind": "operator", "op": n.name,
                               "params": _clean(n.params), "in": list(n.in_ids)})
        else:
            serialized.append({"id": n.id, "kind": "functor", "cls": n.name,
                               "params": _clean(n.params), "in": list(n.in_ids)})
    return {
        "screamer_dag": SCHEMA_VERSION,
        "inputs": [id_to_name[i] for i in input_ids],  # declared call signature order
        "align_outputs": dag.align_outputs,
        "nodes": serialized,
        "outputs": list(output_ids),
    }


def to_json(dag, indent=2):
    return json.dumps(to_dict(dag), indent=indent)


def from_dict(data):
    """Rebuild a runnable ``Dag`` from a dict produced by :func:`to_dict`."""
    version = data.get("screamer_dag")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported screamer_dag version: {version!r} (expected {SCHEMA_VERSION})")
    import screamer  # functor classes live on the top-level namespace (wrapped)

    built = {}  # node id -> rebuilt Node
    for spec in data["nodes"]:
        nid, kind = spec["id"], spec["kind"]
        if kind == "input":
            built[nid] = Input(spec["name"])
        elif kind == "operator":
            op = spec["op"]
            if op not in _OPERATORS:
                raise ValueError(f"unknown DAG operator: {op!r}")
            feeds = [built[i] for i in spec["in"]]
            built[nid] = _OPERATORS[op](*feeds, **spec.get("params", {}))
        elif kind == "functor":
            cls = getattr(screamer, spec["cls"], None)
            if cls is None:
                raise ValueError(f"unknown functor class: {spec['cls']!r}")
            feeds = [built[i] for i in spec["in"]]
            params = dict(spec.get("params", {}))
            args = params.pop("args", [])  # positional fallback for schema-less functors
            built[nid] = cls(*args, **params)(*feeds)
        else:
            raise ValueError(f"unknown node kind: {kind!r}")

    name_to_node = {spec["name"]: built[spec["id"]]
                    for spec in data["nodes"] if spec["kind"] == "input"}
    inputs = [name_to_node[name] for name in data["inputs"]]
    outputs = [built[i] for i in data["outputs"]]
    return Dag(inputs, outputs, align_outputs=data.get("align_outputs", True))


def from_json(text):
    return from_dict(json.loads(text))
