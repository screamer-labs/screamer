---
name: Pipeline
title: Pipeline
kind: graph
short: A reusable N-in / M-out function you define once and call on stored or live data.
tags:
- dag
- graph
- pipeline
- computational graph
- live streaming
topics:
- graphs
covers:
- Input
- Node
---

# `Pipeline`

Define a pipeline once, then run it in batch or live with identical
results. `Input` names a source stream; calling functors and stream operators on
those handles records the pipeline. `Pipeline(inputs=[...], outputs=[...])` compiles it
into a callable: `pipe(arrays)` runs it in batch, `pipe(generators)` runs it event
by event (lazy pull path), and `pipe.live()` opens an incremental session you drive
yourself. All three produce byte-identical output on the same events.

For the conceptual model (how the pipeline is built, when to reach for it, and the
define-once-run-anywhere guarantee), see
[Pipelines](../pipelines.md). This page is the reference contract.

<!-- HELP_END -->

```{eval-rst}
.. autoclass:: screamer.Input
.. autoclass:: screamer.Node
   :special-members: __init__
.. autoclass:: screamer.Pipeline
   :members: live
```

## The three names

- **`Input(name)`**: a function that returns a `Node`, a named source handle.
  The `inputs` list of a `Pipeline` is made of these, and their order and names
  define the pipeline's call signature.
- **`Node`**: the handle type for a stream inside the pipeline. It is user-facing
  as a *type*, but you rarely construct `Node(...)` by hand; you obtain nodes
  from `Input(...)` and by applying functors and stream operators to existing
  nodes.
- **`Pipeline`**: the compiled, runnable object built from `inputs` and `outputs`.

## Constructor

`Pipeline(inputs, outputs, align_outputs=True, max_pending=1_000_000)`

- **`inputs`**: a list of `Input(...)` nodes. Defines the call signature; feeds
  are bound positionally in this order, or by the input names.
- **`outputs`**: a list of nodes to evaluate.
- **`align_outputs`**: with more than one output and `True` (default), every
  output is put on one index: the sorted union of the output indexes, with one
  row per distinct index. Each output holds its latest value at every row, and
  the rows start once every output has a value (an as-of join, as
  `CombineLatest(emit="when_all")`). The outputs then share one index array and
  have equal length; each keeps its own columns, so a 2-column output stays
  `(T, 2)`. When `False`, each output is an independent stream whose length may
  differ from the others.
- **`max_pending`**: the most events a join node may hold while it waits for
  another stream: the reorder buffer of each `CombineLatest` and `Filter` node
  (also the join of a threshold `Resample` with its driver). More raises
  `RuntimeError` and ends the call or session, so a stalled stream cannot grow
  memory without bound. An integer from 1 to `2**53`; another number raises
  `ValueError`, and a value that is not an integer `TypeError`.

The constructor validates the pipeline and raises a clear `ValueError` if: an
`inputs` entry is not an `Input(...)` node; an `outputs` entry is not a `Node`;
an output references an undeclared input; a declared input is never used; or a
single functor instance backs more than one node.

## Calling the pipeline

Feeds are passed positionally in input order, `pipe(*feeds)`, or by input name,
`pipe(**named_feeds)`. Each feed may be:

- a bare array (positional, with the index taken as the row number), or
- a `(values, index)` pair.

The return is always `(values, index)` tuples:

- **one output**, a single `(values, index)` pair,
- **multiple outputs**, a tuple of such pairs, one per output.

Pass generators of `(value, index)` pairs instead of arrays to run the pipeline
lazily, event by event: `pipe(gen_a, gen_b)` returns an iterator that yields
output events byte-identical to the batch result.

A pipeline stream holds flat rows of numbers. An operator that gives several
values per column flattens them: `Resample(agg="ohlc")` on an `X`-column
stream gives `(T', 4 * X)` in column-major field order,
`[open_0, high_0, low_0, close_0, open_1, ...]`, where the eager call gives
`(T', X, 4)`. Reshape to recover the eager result:

```python
x = Input("x")
pipe = Pipeline([x], [Resample(freq=60, agg="ohlc")(x)])
values, index = pipe((panel, t))               # (T', 4 * X)
bars = values.reshape(len(values), -1, 4)      # (T', X, 4), as the eager call
```

Phase 2 of the shape contract (asset axes inside a `Pipeline`) will return
`(T', X, 4)` directly.

An output with no rows keeps its width when the graph or the feeds fix it: an
empty `(0, 3)` feed through the pipeline above gives `(0, 12)` values, and an
operator with three outputs gives `(0, 3)` even before any event. Only a
one-column output, or one whose width depends on a live input that has had no
event yet, is `(0,)`. The same holds for each `live().result()`, so empty
results concatenate with the others.

## Live, incremental sessions: `pipe.live()`

`pipe(arrays)` and `pipe(generators)` both consume complete feeds. When you drive
the pipeline yourself, one event at a time, `pipe.live()` opens a session object. It
shares the pipeline's single engine and resets it on open, so use one
session at a time (do not interleave it with a `pipe(...)` call). Each method
returns the session, so calls chain.

- **`.push(input, index, value)`** feeds one event. `input` is an `Input` name
  (str) or its position (int) in the `inputs` list; `index` is the event's integer
  index; `value` is the float value. An index below the input's previous event,
  or below an earlier `.advance(now)`, raises `ValueError` and the session goes on
  as if the event had not been sent. Equal indexes are allowed.
- **`.advance(now)`** moves logical time to `now` (an integer index), closing every
  windowing node whose bucket boundary has passed by `now`. This finalizes
  time-based bars even when no event fell in them, for example an empty minute bar
  closed by a clock tick. It is a no-op for event-count (`count=`) windows and
  before the first event; call it with non-decreasing `now`.
- **`.flush()`** finalizes the current partial window(s) on demand, for example at
  the end of a processing loop. The end-of-input flush that `pipe(...)` performs
  implicitly is the special case. Use `.flush()`, not `advance`, to end the input:
  `advance` leaves `count=` and `threshold=` windows open.
- **`.result()`** returns the output accumulated so far, in the same shape
  `pipe(...)` returns, and drains the internal buffers. Aligned outputs keep
  each output's latest value from one call to the next, so the rows of repeated
  calls, put end to end, are the rows of one call at the end. A call returns a
  row only once every output has moved past its index (another output could
  still change it); after `.flush()` it returns every row.

An input that gets no events holds the aligned rows. While an input is silent,
for example a clock that stopped ticking or a feed that paused, the outputs fed
by it do not move, so the other outputs' rows wait in memory until
`.advance(now)` or `.flush()`. In a long-running session, call `.advance(now)`
from time to time: it moves every input to `now` and releases the rows below it.

Feeding the same events in index order and then calling `.flush()` reproduces the
batch result exactly. `.advance()` (and a clock input wired into the pipeline)
additionally let a windowing node emit bars a purely event-driven pass would not,
the empty leading and trailing bars in
[`Resample`](../functions_streams/Resample.md), for instance.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Input, Pipeline, First, Last, Resample, CombineLatest

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

## Examples

Align two streams and take their difference. The same pipeline runs in batch and
live.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Input, Pipeline, Sub, CombineLatest

   a, b = Input("a"), Input("b")
   pipe = Pipeline(inputs=[a, b], outputs=[Sub()(CombineLatest()(a, b))])

   # feeds are (values, index), values first
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

## Inspecting a pipeline

`print(pipe)` (or `pipe.to_text()`) shows the pipeline as an indented tree, rooted at
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

   from screamer import Input, Pipeline, RollingMean, Sub, CombineLatest

   a, b = Input("a"), Input("b")
   pipe = Pipeline(inputs=[a, b], outputs=[RollingMean(20)(Sub()(CombineLatest()(a, b)))])
   print(pipe.to_text())
```

## Saving and loading a pipeline

`pipe.to_json()` serializes the pipeline to JSON (its inputs, nodes with their
parameters, outputs, `align_outputs` and `max_pending`), and
`Pipeline.from_json(text)` rebuilds a runnable `Pipeline` from it. This
round-trips exactly, so a pipeline can be saved as a config file and reloaded.
`to_dict` / `from_dict` give the same round-trip with a plain dict. A functor
reducer of a `Resample`, as in `agg=RollingMean(5)`, is saved as its name and
parameters, `{"functor": "RollingMean", "params": {"window_size": 5}}`, and
loading builds a new one, so the restored pipeline shares no state with the
original.

The format is shared with JavaScript: a pipeline saved by `to_json()` loads
with `Pipeline.fromJSON(text)` in `@screamer-labs/screamer`, and one saved by
its `toJSON()` loads here, with the same results. Parameters keep their
Python names in both languages (`window_size`, not `windowSize`). A setting
that is not finite, such as `Clip(lower=-inf)`, is written as `-Infinity`,
which Python's `json` module reads; strict JSON readers do not. `from_json`
and `from_dict` check the description first and raise `ValueError` naming
the problem, such as "unknown functor class: 'NoSuchOp'" or "Pipeline
description: node 1: input 7 is not an earlier node"; each node's settings
are then checked by its constructor.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Input, Pipeline, RollingMean, Sub, CombineLatest

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
  pipeline relies on.
