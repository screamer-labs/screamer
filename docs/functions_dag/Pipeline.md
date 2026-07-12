---
name: Pipeline
title: Pipeline
kind: graph
short: A reusable N-in / M-out function you define once and call on stored or live data.
topics:
- graphs
covers:
- Input
- Node
---

# `Pipeline`

Define a computation graph once, then run it in batch or live with identical
results. `Input` names a source stream; calling functors and stream operators on
those handles records the graph. `Pipeline(inputs=[...], outputs=[...])` compiles it
into a callable: `pipe(arrays)` runs it in batch, `pipe(generators)` runs it event
by event (lazy pull path), and `pipe.live()` opens an incremental session you drive
yourself. All three produce byte-identical output on the same events.

For the conceptual model (how the graph is built, when to reach for it, and the
define-once-run-anywhere guarantee), see
[Pipelines](../pipelines.md). This page is the reference contract.

<!-- HELP_END -->

```{eval-rst}
.. autoclass:: screamer.dag.Input
.. autoclass:: screamer.dag.Node
   :special-members: __init__
.. autoclass:: screamer.dag.Pipeline
   :members: live
```

## The three names

- **`Input(name)`**: a function that returns a `Node`, a named source handle.
  The `inputs` list of a `Pipeline` is made of these, and their order and names
  define the graph's call signature.
- **`Node`**: the handle type for a stream inside the graph. It is user-facing
  as a *type*, but you rarely construct `Node(...)` by hand; you obtain nodes
  from `Input(...)` and by applying functors and stream operators to existing
  nodes.
- **`Pipeline`**: the callable compiled graph.

## Constructor

`Pipeline(inputs, outputs, align_outputs=True)`

- **`inputs`**: a list of `Input(...)` nodes. Defines the call signature; feeds
  are bound positionally in this order, or by the input names.
- **`outputs`**: a list of nodes to evaluate.
- **`align_outputs`**: when `True` (default), all outputs are co-indexed onto a
  single shared index, so every output is an equal-length `(values, index)`
  pair. When `False`, each output is returned as an independent stream whose
  length may differ from the others.

The constructor validates the graph and raises a clear `ValueError` if: an
`inputs` entry is not an `Input(...)` node; an `outputs` entry is not a `Node`;
an output references an undeclared input; a declared input is never used; or a
single functor instance backs more than one node.

## Calling the graph

Feeds are passed positionally in input order, `pipe(*feeds)`, or by input name,
`pipe(**named_feeds)`. Each feed may be:

- a bare array (positional, with the index taken as the row number), or
- a `(values, index)` pair.

The return is always `(values, index)` tuples:

- **one output**, a single `(values, index)` pair,
- **multiple outputs**, a tuple of such pairs, one per output.

Pass generators of `(value, index)` pairs instead of arrays to run the graph
lazily, event by event: `pipe(gen_a, gen_b)` returns an iterator that yields
output events byte-identical to the batch result.

## Live, incremental sessions: `pipe.live()`

`pipe(arrays)` and `pipe(generators)` both consume complete feeds. When you drive
the graph yourself, one event at a time, `pipe.live()` opens a session object. It
shares the compiled graph's single engine and resets it on open, so use one
session at a time (do not interleave it with a `pipe(...)` call). Each method
returns the session, so calls chain.

- **`.push(input, index, value)`** feeds one event. `input` is an `Input` name
  (str) or its position (int) in the `inputs` list; `index` is the event's integer
  index; `value` is the float value.
- **`.advance(now)`** moves logical time to `now` (an integer index), closing every
  windowing node whose bucket boundary has passed by `now`. This finalizes
  time-based bars even when no event fell in them, for example an empty minute bar
  closed by a clock tick. It is a no-op for event-count (`count=`) windows and
  before the first event; call it with non-decreasing `now`.
- **`.flush()`** finalizes the current partial window(s) on demand, for example at
  the end of a processing loop. The end-of-input flush that `pipe(...)` performs
  implicitly is the special case.
- **`.result()`** returns the output accumulated so far, in the same shape
  `pipe(...)` returns, and drains the internal buffers.

Feeding the same events in index order and then calling `.flush()` reproduces the
batch result exactly. `.advance()` (and a clock input wired into the graph)
additionally let a windowing node emit bars a purely event-driven pass would not,
the empty leading and trailing bars in
[`Resample`](../functions_streams/Resample.md), for instance.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Input, Pipeline, First, Last, Resample, CombineLatest
   # --- hide: stop ---
   price = Input("price")
   open_b  = Resample(freq=10, fill="nan", agg=First())(price)
   close_b = Resample(freq=10, fill="nan", agg=Last())(price)
   bars = CombineLatest()(open_b, close_b)
   pipe = Pipeline(inputs=[price], outputs=[bars])

   live = pipe.live()
   live.push("price", 0, 100.0)   # one trade in bar [0, 10)
   live.advance(30)               # clock passes bars [10,20) and [20,30): empty -> NaN
   live.flush()
   values, index = live.result()
   print(index)
   print(values)
```

## Example

Align two streams and take their difference. The same graph runs in batch and
live.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Input, Pipeline, Sub, CombineLatest
   # --- hide: stop ---
   a, b = Input("a"), Input("b")
   pipe = Pipeline(inputs=[a, b], outputs=[Sub()(CombineLatest()(a, b))])

   # feeds are (values, index) - values-first
   fa = (np.array([10.0, 20.0, 30.0]), np.array([1, 2, 3]))
   fb = (np.array([1.0, 2.0, 3.0]),   np.array([1, 2, 3]))

   spread, idx = pipe(fa, fb)
   print(spread.reshape(-1))

   # pipe(generators) returns the identical result, event by event (lazy path)
   events = list(pipe(
       ((v, k) for v, k in zip(fa[0], fa[1])),
       ((v, k) for v, k in zip(fb[0], fb[1])),
   ))
   sv = np.array([e[0] for e in events])
   print(np.array_equal(spread.reshape(-1), sv, equal_nan=True))
```

## Inspecting a graph

`print(pipe)` (or `pipe.to_text()`) shows the graph as an indented tree, rooted at
each output and descending to the inputs. A node shared by several consumers is
printed once and then referenced by id, so a diamond reads as a diamond. Node
labels carry the functor and operator parameters (`RollingMean(window_size=20)`,
`Resample(every=5, ...)`).

For a diagram, `pipe.to_dot()` returns a Graphviz DOT string with no dependencies,
and `pipe.to_graphviz()` returns a rendered `graphviz.Source` when the optional
`graphviz` package is installed (`pip install screamer[viz]`, plus the system
Graphviz `dot` binary). In a Jupyter notebook, displaying a `Pipeline` shows the diagram
inline, falling back to the text tree when graphviz is not available.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   from screamer import Input, Pipeline, RollingMean, Sub, CombineLatest
   # --- hide: stop ---
   a, b = Input("a"), Input("b")
   pipe = Pipeline(inputs=[a, b], outputs=[RollingMean(20)(Sub()(CombineLatest()(a, b)))])
   print(pipe.to_text())
```

## Saving and loading a graph

`pipe.to_json()` serializes the graph to JSON (its inputs, nodes with their
parameters, outputs, and `align_outputs`), and `Pipeline.from_json(text)` rebuilds a
runnable `Pipeline` from it. This round-trips exactly, so a graph can be saved as a
config file and reloaded. `to_dict` / `from_dict` give the same round-trip with a
plain dict.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Input, Pipeline, RollingMean, Sub, CombineLatest
   # --- hide: stop ---
   a, b = Input("a"), Input("b")
   pipe = Pipeline(inputs=[a, b], outputs=[RollingMean(20)(Sub()(CombineLatest()(a, b)))])

   config = pipe.to_json()          # save this string to a file
   restored = Pipeline.from_json(config)

   fa = (np.arange(1.0, 61), np.arange(1, 61))
   fb = (np.arange(0.0, 60), np.arange(1, 61))
   print(np.array_equal(pipe(fa, fb)[0], restored(fa, fb)[0], equal_nan=True))
```

## See also

- [Pipelines](../pipelines.md): the conceptual model and walkthrough.
- [Streams, values, and alignment](../multistream.md): the alignment model the
  graph relies on.
