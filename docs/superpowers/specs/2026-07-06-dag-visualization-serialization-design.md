# DAG visualization and serialization — design

**Date:** 2026-07-06
**Status:** approved design, pending spec review
**Topic:** inspect a `Dag` (text + graphviz) and round-trip it to JSON

## Goal

Make a `Dag` inspectable and portable:
1. **See it** — a zero-dependency ASCII view and an optional graphviz diagram, with
   auto-display in Jupyter.
2. **Serialize it** — dump a `Dag` to JSON and rebuild an identical one, so a graph
   can be saved as a config file.

Both rest on one new capability: capturing each functor's constructor arguments.

## Decisions made during brainstorming

- **Graph backend:** graphviz as an *optional* dependency. `to_dot()` always
  returns a DOT string; `to_graphviz()` and the Jupyter repr render via the
  `graphviz` package when installed, else fall back to the ASCII view. (Alternatives
  mermaid / both were considered; graphviz is the standard for compute graphs and
  auto-renders inline in notebooks.)
- **Functor labels:** invest in real params now (`RollingMean(window_size=20)`), not
  class-name-only.
- **Param-capture mechanism:** wrap each functor in a thin, same-named **Python
  subclass** whose `__init__` records its arguments. Verified: such a subclass still
  builds a graph node and runs correctly through the compiled C++ engine, and
  `isinstance` against the C++ class still holds. This is ~40 lines of generic code
  driven by the `help.json` parameter schema, versus adding accessors to ~150 C++
  bindings. Accepted implication: `from screamer import RollingMean` returns a
  same-named subclass (transparent: identical name, same C++ compute, a small
  `_screamer_params` dict per instance; construction goes through a Python
  `__init__`, negligible since it is not per-sample).
- **What to capture:** the arguments the caller actually passed, bound to parameter
  names via the schema. This reconstructs identically and stays compact (callers
  normally pass only non-defaults); no diffing against defaults is required.

## Component 1: functor parameter capture

New module `screamer/_functor_params.py`:

- `bind_params(cls_name, args, kwargs) -> dict`: bind positional `args` to parameter
  names using the `help.json` schema for `cls_name` (which lists parameters in
  order), merge `kwargs`. Functors absent from `help.json` fall back to
  `{"args": [...], "kwargs": {...}}` positional capture.
- `install_param_capture(namespace)`: for every name in `namespace` bound to a
  `ScreamerBase` subclass (excluding the `ScreamerBase` / `EvalOp` base types),
  replace it with a subclass:

  ```python
  class RollingMean(_RollingMean):          # same __name__
      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)  # constructs + validates in C++
          self._screamer_params = bind_params("RollingMean", args, kwargs)
      def __repr__(self):
          return _format_call("RollingMean", self._screamer_params)
  ```

  This also gives functors a readable `repr` (`RollingMean(window_size=20)`) in place
  of the default object address.

**Integration:** `devtools/generate_screamer__init__.py` appends, after the existing
imports, `from ._functor_params import install_param_capture` and
`install_param_capture(globals())`. So regenerating `__init__.py` keeps the wrapping.
The raw C++ classes remain reachable via `screamer.screamer_bindings` if ever needed.

## Component 2: shared graph model

A single traversal (in `screamer/dag_viz.py`, reused by serialization) walks the
`Dag`'s outputs, dedups nodes by `id`, and returns an ordered node list
(dependencies before dependents) where each entry has: a stable integer id, a kind
(`input` / `operator` / `functor`), a label, a params dict, and the ids of its
inputs. Inputs and outputs are tagged. This model backs `to_text`, `to_dot`, and
`to_dict` so labelling and structure stay consistent.

Label rules (from the brainstorm feasibility check):
- input → the name (`a`)
- operator → `name(params)` from the node's stored kwargs (`resample(every=5)`)
- functor → `Class(params)` from `_screamer_params` (`RollingMean(window_size=20)`)

## Component 3: visualization API (on `Dag`)

- `dag.to_text() -> str`, and `str(dag)` / `print(dag)` — an indented `tree`-style
  view rooted at each output, descending to inputs. A node shared by several
  consumers is printed once and later referenced by id (e.g. `└─ ↑#3 Sub`) so a
  diamond reads as a diamond rather than being duplicated. `repr(dag)` stays a
  one-liner (`Dag(2 inputs, 1 output)`).
- `dag.to_dot() -> str` — a Graphviz DOT string (zero dependency). `rankdir=LR`;
  inputs and outputs styled distinctly from operators/functors; edges point
  upstream→downstream (data flow).
- `dag.to_graphviz()` — returns a `graphviz.Digraph` (renders to SVG/PNG). Raises a
  clear `ImportError` with an install hint if `graphviz` is not installed;
  `to_dot()` still works without it.
- `dag._repr_mimebundle_(...)` — inline SVG in Jupyter when `graphviz` is present,
  else the ASCII text. So typing `dag` in a notebook shows the diagram.

## Component 4: serialization API (on `Dag`)

- `dag.to_dict() -> dict`, `dag.to_json(indent=2) -> str`
- `Dag.from_dict(d) -> Dag`, `Dag.from_json(s) -> Dag` (classmethods)

Schema (version-tagged):

```json
{
  "screamer_dag": 1,
  "inputs": ["a", "b"],
  "align_outputs": true,
  "nodes": [
    {"id": 0, "kind": "input", "name": "a"},
    {"id": 1, "kind": "input", "name": "b"},
    {"id": 2, "kind": "functor",  "cls": "RollingMean", "params": {"window_size": 20}, "in": [0]},
    {"id": 3, "kind": "operator", "op": "combine_latest", "params": {"emit": "when_all"}, "in": [2, 1]},
    {"id": 4, "kind": "functor",  "cls": "Sub", "params": {}, "in": [3]}
  ],
  "outputs": [4]
}
```

Reconstruction (`from_dict`): build an id→Node map in list order —
- `input` → `Input(name)`
- `operator` → `OPERATORS[op](*[map[i] for i in in], **params)` where `OPERATORS`
  maps the four graph operators (`combine_latest`, `dropna`, `select`, `resample`)
  to their `streams` functions
- `functor` → `getattr(screamer, cls)(**params)(*[map[i] for i in in])`

then `Dag(inputs=[map[i] for input nodes], outputs=[map[i] for outputs], align_outputs)`.

All params are JSON-native (ints, floats, strings, and `MovingAverage`'s tap list).
Validation on load: check `screamer_dag` version; unknown `cls`/`op` → clear error;
malformed edges → clear error.

## Reusable elements / integration points

- `help.json` (already loaded from `screamer/data/help.json`) supplies parameter
  names for positional binding.
- The four graph operators come from the existing `_compile_cpp` support list
  (`combine_latest`, `dropna`, `select`, `resample`); the serialization registry
  mirrors it exactly, so both evolve together.
- `pyproject.toml`: add a `viz = ["graphviz"]` entry under the existing
  `[project.optional-dependencies]`. Note in docs that graphviz also needs the
  system `dot` binary for rendering.

## Code layout

- `screamer/_functor_params.py` — capture + wrapping.
- `screamer/dag_viz.py` — graph model, `to_text`, `to_dot`, `to_graphviz`, repr.
- `screamer/dag_io.py` — `to_dict` / `from_dict` / json + operator registry.
- `screamer/dag.py` — thin `Dag` methods delegating to the two modules.
- `devtools/generate_screamer__init__.py` — emit the wrapping hook.

## Testing

- `tests/test_functor_params.py`: params captured for a spread of functors
  (positional, keyword, mixed; no-arg like `Sub`; list arg like `MovingAverage`);
  `repr` reads correctly; wrapped functors still run and pass through the DAG engine;
  `isinstance` against the C++ class holds.
- `tests/test_dag_viz.py`: `to_text` golden output for a graph with a diamond and
  two outputs (shared node referenced, not duplicated); `to_dot` contains the
  expected labels/edges and parses; `to_graphviz` skipped when graphviz is absent;
  Jupyter repr returns SVG or text.
- `tests/test_dag_io.py`: **round-trip equivalence** is the core test — build a
  non-trivial graph (functors with params, all four operators, a multi-input functor,
  multiple outputs, `align_outputs` True and False, a shared node), `to_json` →
  `from_json`, run both on the same feeds, assert identical results; unknown
  class/operator and bad version raise clear errors.

## Out of scope / future

- Visualizing a bare `Node` subgraph before it is wrapped in a `Dag`.
- Mermaid export.
- Capturing functor params via C++ accessors (the heavier alternative we did not take).
- Serializing feeds/data — only the graph structure is serialized.

## Open items for the plan

- Exact ASCII glyphs and the shared-node reference format.
- Whether `to_json` sorts nodes topologically or by discovery order (both valid;
  pick one for stable output).
- `graphviz` node/edge styling specifics (shapes, colors).
