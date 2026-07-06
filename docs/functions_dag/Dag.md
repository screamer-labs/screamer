---
name: Dag
title: Dag
kind: graph
short: A positional N-in / M-out callable that evaluates a computation graph.
topics:
- graphs
covers:
- Input
- Node
---

# `Dag`

Define a computation graph once, then run it in batch or live with identical
results. `Input` names a source stream; calling functors and stream operators on
those handles records the graph. `Dag(inputs=[...], outputs=[...])` compiles it
into a callable: `dag(...)` runs it on stored arrays, `dag.stream(...)` runs it
live, event by event, and the two produce byte-identical output.

For the conceptual model — how the graph is built, when to reach for it, and the
define-once-run-anywhere guarantee — see
[The computational graph](../dag.md). This page is the reference contract.

<!-- HELP_END -->

```{eval-rst}
.. autoclass:: screamer.dag.Input
.. autoclass:: screamer.dag.Node
   :special-members: __init__
.. autoclass:: screamer.dag.Dag
   :members: stream
```

## The three names

- **`Input(name)`** — a function that returns a `Node`: a named source handle.
  The `inputs` list of a `Dag` is made of these, and their order and names
  define the graph's call signature.
- **`Node`** — the handle type for a stream inside the graph. It is user-facing
  as a *type*, but you rarely construct `Node(...)` by hand; you obtain nodes
  from `Input(...)` and by applying functors and stream operators to existing
  nodes.
- **`Dag`** — the callable compiled graph.

## Constructor

`Dag(inputs, outputs, align_outputs=True)`

- **`inputs`** — a list of `Input(...)` nodes. Defines the call signature: feeds
  are bound positionally in this order, or by the input names.
- **`outputs`** — a list of nodes to evaluate.
- **`align_outputs`** — when `True` (default), all outputs are co-indexed onto a
  single shared index, so every output is an equal-length `(values, index)`
  pair. When `False`, each output is returned as an independent stream whose
  length may differ from the others.

The constructor validates the graph and raises a clear `ValueError` if: an
`inputs` entry is not an `Input(...)` node; an `outputs` entry is not a `Node`;
an output references an undeclared input; a declared input is never used; or a
single functor instance backs more than one node.

## Calling the graph

Feeds are passed positionally in input order, `dag(*feeds)`, or by input name,
`dag(**named_feeds)`. Each feed may be:

- a bare array — positional, with the index taken as the row number,
- a `Stream`, or
- a `(values, index)` pair.

The return is always `(values, index)` tuples — never `Stream` objects, even
when you feed `Stream`s:

- **one output** → a single `(values, index)` pair,
- **multiple outputs** → a tuple of such pairs, one per output.

`dag.stream(*feeds)` accepts the same feeds and returns results byte-identical to
`dag(*feeds)`; only the execution mode differs (event-by-event rather than in one
batch pass).

## Example

Align two streams and take their difference. The same graph runs in batch and
live.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Input, Dag, Sub
   from screamer.streams import combine_latest
   # --- hide: stop ---
   a, b = Input("a"), Input("b")
   dag = Dag(inputs=[a, b], outputs=[Sub()(combine_latest(a, b))])

   # feeds are (values, index) - values-first
   fa = (np.array([10.0, 20.0, 30.0]), np.array([1, 2, 3]))
   fb = (np.array([1.0, 2.0, 3.0]),   np.array([1, 2, 3]))

   spread, idx = dag(fa, fb)
   print(spread.reshape(-1))

   # dag.stream(fa, fb) returns the identical result, event by event
   sv, sk = dag.stream(fa, fb)
   print(np.array_equal(spread, sv, equal_nan=True))
```

## Inspecting a graph

`print(dag)` (or `dag.to_text()`) shows the graph as an indented tree, rooted at
each output and descending to the inputs. A node shared by several consumers is
printed once and then referenced by id, so a diamond reads as a diamond. Node
labels carry the functor and operator parameters (`RollingMean(window_size=20)`,
`resample(every=5)`).

For a diagram, `dag.to_dot()` returns a Graphviz DOT string with no dependencies,
and `dag.to_graphviz()` returns a rendered `graphviz.Source` when the optional
`graphviz` package is installed (`pip install screamer[viz]`, plus the system
Graphviz `dot` binary). In a Jupyter notebook, displaying a `Dag` shows the diagram
inline, falling back to the text tree when graphviz is not available.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   from screamer import Input, Dag, RollingMean, Sub
   from screamer.streams import combine_latest
   # --- hide: stop ---
   a, b = Input("a"), Input("b")
   dag = Dag(inputs=[a, b], outputs=[RollingMean(20)(Sub()(combine_latest(a, b)))])
   print(dag.to_text())
```

## Saving and loading a graph

`dag.to_json()` serializes the graph to JSON (its inputs, nodes with their
parameters, outputs, and `align_outputs`), and `Dag.from_json(text)` rebuilds a
runnable `Dag` from it. This round-trips exactly, so a graph can be saved as a
config file and reloaded. `to_dict` / `from_dict` give the same round-trip with a
plain dict.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Input, Dag, RollingMean, Sub
   from screamer.streams import combine_latest
   # --- hide: stop ---
   a, b = Input("a"), Input("b")
   dag = Dag(inputs=[a, b], outputs=[RollingMean(20)(Sub()(combine_latest(a, b)))])

   config = dag.to_json()          # save this string to a file
   restored = Dag.from_json(config)

   fa = (np.arange(1.0, 61), np.arange(1, 61))
   fb = (np.arange(0.0, 60), np.arange(1, 61))
   print(np.array_equal(dag(fa, fb)[0], restored(fa, fb)[0], equal_nan=True))
```

## See also

- [The computational graph](../dag.md) — the conceptual model and walkthrough.
- [Streams, values, and alignment](../multistream.md) — the alignment model the
  graph relies on.
