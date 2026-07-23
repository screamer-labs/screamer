# Pipelines

A `Pipeline` wires functors and stream operators into a reusable computation you
define once and run either on stored data or live, event by event. The same
guarantee that holds for a single functor - causal output, no lookahead - holds
for a whole pipeline built from them.

This page explains the model and when to reach for it. For the exact
constructor, feed forms, and return shapes, see the [`Pipeline` reference](functions_dag/Pipeline.md);
for a full worked walkthrough, see the pipelines example notebook.

## When you need a Pipeline (and when you don't)

For a single stream flowing through a chain of functors you don't need a `Pipeline`.
Ordinary composition is enough:

```python
signal = Sign()(RollingPoly2(50, derivative_order=1)(data))
```

A `Pipeline` earns its place when the pipeline is more than a single chain:

- **several inputs that don't tick together**: two price feeds on different
  clocks that must be aligned before they are combined,
- **several outputs**: you want a raw spread *and* its smoothed signal from one
  pass, co-indexed onto a shared timeline,
- **one deployable object**: you want to hand the whole pipeline around, test it
  on history, and run it live without re-wiring it.

In short: reach for a `Pipeline` when you would otherwise be threading alignment and
multiple streams together by hand.

## The model: inputs, nodes, and the pipeline

Three names make up the surface, and they play distinct roles:

- **`Input(name)`** marks a source: a named place where a stream will enter the
  pipeline. It returns a node.
- **A node** is a *handle* to a stream inside the pipeline, not the data itself.
  You get the first nodes from `Input(...)`, then build more by applying functors
  and stream operators to existing nodes. Applying an operator to a node returns
  a new node.
- **`Pipeline(inputs=[...], outputs=[...])`** compiles the nodes reachable from those
  outputs into a single callable.

Building the pipeline only *records structure*; nothing computes until you call the
compiled `Pipeline`. Applying `RollingMean(10)` to a node does not run a rolling mean;
it adds a node that *will* run one when data flows.

```python
from screamer import Input, Pipeline, RollingMean, Sub, CombineLatest

a, b = Input("a"), Input("b")         # two named sources
spread = Sub()(CombineLatest()(a, b)) # align, then difference -> a node
signal = RollingMean(10)(spread)      # smooth it -> another node

pipe = Pipeline(inputs=[a, b], outputs=[signal])   # compile the pipeline
```

## Running a pipeline

The pipeline runs in two modes:

- **Batch**: `pipe(feed_a, feed_b)` evaluates the whole pipeline over stored arrays
  in one pass and returns `(values, index)` arrays.
- **Lazy (event-by-event)**: pass generators of `(value, index)` pairs instead
  of arrays, `pipe(gen_a, gen_b)`, and the pipeline runs event by event, pulling one
  event at a time. The result is an iterator that yields output events; values are
  byte-identical to the batch result.

Both forms use the same C++ engine. Feeds can be bare arrays, `(values, index)` pairs, or generators of `(value, index)` pairs.

(pipeline-live)=
### Incremental, clock-driven: `pipe.live()`

When you drive the pipeline yourself,
event by event, `pipe.live()` opens a session:

- `.push(input, index, value)` feeds one event (`input` is an `Input` name or its
  position);
- `.advance(now)` moves logical time to `now`, closing every window whose boundary
  has passed (this finalizes time-based bars even when no new data arrives, such as
  an empty minute bar closed by a clock tick);
- `.flush()` finalizes the current partial window on demand, for example at the end
  of a loop;
- `.result()` collects the aligned output so far.

Feeding the same events and calling `.flush()` reproduces the batch result.
`.advance()` (and a clock input wired into the pipeline) additionally let a windowing
node emit bars that a purely event-driven pass would not, such as the empty leading
and trailing bars in [`Resample`](functions_streams/Resample.md).

## Multiple outputs and alignment

A pipeline can expose more than one output, for example the raw spread and its
smoothed signal together. By default (`align_outputs=True`) all outputs are
co-indexed onto one shared timeline, so every output is an equal-length
`(values, index)` pair and column `k` of one lines up with column `k` of
another. Set `align_outputs=False` when you want each output as an independent
stream whose length may differ. The precise semantics are in the
[reference](functions_dag/Pipeline.md).

## The guarantees

- **Causal.** Every node's output at index `t` depends only on events at indices
  `<= t`. Wiring functors into a pipeline does not introduce any lookahead.
- **Mode-consistent.** `pipe(arrays)` and `pipe(generators)` emit the same values
  in the same order; only the execution mode differs. The alignment a pipeline
  performs is causal and identical whether you run on a stored dataset or feed
  events one at a time.

These are the same guarantees the single-functor API and the stream layer make;
the `Pipeline` simply preserves them across a whole composition.

## See also

- [`Pipeline` reference](functions_dag/Pipeline.md): constructor, feed forms, return
  shapes, and validation rules.
- [Streams, values, and alignment](multistream.md): the alignment model that
  `CombineLatest` and friends bring into a pipeline.
