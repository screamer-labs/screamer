"""Serialize a ``Pipeline`` to a JSON-native dict (and back).

A Pipeline is captured as its ordered input names, its nodes (each input / operator /
functor with params and the ids of the nodes feeding it), its outputs, and
``align_outputs`` and ``max_pending``. ``from_dict`` rebuilds a runnable Pipeline, so a graph round-trips
as a config file. Reuses the same graph walk as the visualizer so structure and
labels stay consistent.

The schema and the JSON text are C++ (include/screamer/dag/graph_json.h),
shared with JavaScript's ``Pipeline.toJSON`` and ``Pipeline.fromJSON``: a
description is checked there and given in its normal form, and this module
builds the nodes from it with the Python constructors.
"""
import numpy as np

from . import screamer_bindings as _b

from . import streams
from .dag import Pipeline, Input
from .dag_viz import build_graph
from .supervised import forecast_pairs as _forecast_pairs

SCHEMA_VERSION = 1

# Operators that can be graph nodes (mirrors Pipeline._compile_cpp's support),
# by the canonical names of the schema (graph_json.h), which also reads the
# lowercase names of graphs saved before step 3E.
_OPERATORS = {
    "CombineLatest": streams.combine_latest,
    "Dropna": streams.dropna,
    "Select": streams.select,
    "Resample": streams.resample,
    "Filter": lambda data, mask: streams.Filter()(data, mask),
    "Delay": lambda values, duration: streams.delay(values, duration=duration),
    "forecast_pairs": _forecast_pairs,
}


def _jsonable(value):
    """Coerce a captured param value to a JSON-native type.

    Constructor args are stored verbatim, so a common ``RollingMean(arr.size // 10)``
    yields a numpy scalar. numpy scalars/arrays are cast to Python ints/floats/lists.
    A functor (a Resample reducer, ``agg=RollingMean(5)``) becomes
    ``{"functor": name, "params": {...}}``, which :func:`from_dict` builds anew.
    """
    if getattr(type(value), "_screamer_wrapped", False):
        return {"functor": type(value).__name__,
                "params": _clean(getattr(value, "_screamer_params", {}))}
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _clean(params):
    """Drop None-valued params and coerce the rest to JSON-native types."""
    return {k: _jsonable(v) for k, v in params.items() if v is not None}


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
        "max_pending": dag.max_pending,
        "nodes": serialized,
        "outputs": list(output_ids),
    }


def to_json(dag, indent=2):
    """The JSON text of :func:`to_dict`, as ``json.dumps(to_dict(dag), indent=indent)``
    writes it (written by the shared C++ writer, which JavaScript uses too)."""
    return _b._graph_to_json(to_dict(dag), indent)


def _functor(spec):
    """A new functor from its name and params."""
    import screamer
    cls = getattr(screamer, spec["functor"], None) if isinstance(spec["functor"], str) else None
    if cls is None or not getattr(cls, "_screamer_wrapped", False):
        _b._unknown_functor(str(spec["functor"]))
    params = {k: _decode(v) for k, v in dict(spec.get("params", {})).items()}
    args = params.pop("args", [])
    return cls(*args, **params)


def _decode(value):
    """A param value as to_dict wrote it, with each functor built anew."""
    if isinstance(value, dict):
        if set(value) <= {"functor", "params"} and "functor" in value:
            return _functor(value)
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def from_dict(data):
    """Rebuild a runnable ``Pipeline`` from a dict produced by :func:`to_dict`."""
    return _build(_b._graph_check(data))


def _build(data):
    """The Pipeline of a description in its normal form (graph_json.h
    check_graph): canonical operator names, params on every node."""
    import screamer  # functor classes live on the top-level namespace (wrapped)

    built = {}  # node id -> rebuilt Node
    for k, spec in enumerate(data["nodes"]):
        nid, kind = spec["id"], spec["kind"]
        if kind == "input":
            built[nid] = Input(spec["name"])
            continue
        feeds = [built[i] for i in spec["in"]]
        params = {k: _decode(v) for k, v in spec["params"].items()}
        if kind == "operator":
            built[nid] = _OPERATORS[spec["op"]](*feeds, **params)
        else:
            cls = getattr(screamer, spec["cls"], None)
            if cls is None or not getattr(cls, "_screamer_wrapped", False):
                _b._unknown_functor(spec["cls"])
            args = params.pop("args", [])  # positional fallback for schema-less functors
            op = cls(*args, **params)
            _b._check_functor_inputs(k, spec["cls"], op.num_inputs, len(feeds))
            built[nid] = op(*feeds)

    name_to_node = {spec["name"]: built[spec["id"]]
                    for spec in data["nodes"] if spec["kind"] == "input"}
    inputs = [name_to_node[name] for name in data["inputs"]]
    outputs = [built[i] for i in data["outputs"]]
    kwargs = {"align_outputs": data["align_outputs"]}
    if "max_pending" in data:              # absent in graphs saved before 3.0.0
        kwargs["max_pending"] = data["max_pending"]
    return Pipeline(inputs, outputs, **kwargs)


def from_json(text):
    """Rebuild a runnable ``Pipeline`` from the text :func:`to_json` writes."""
    if not isinstance(text, str):
        raise TypeError(f"Pipeline.from_json takes the JSON text (a str), got {type(text).__name__}")
    return _build(_b._graph_from_json(text))
