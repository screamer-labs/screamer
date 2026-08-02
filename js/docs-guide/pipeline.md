# Pipelines and graphs

A single op is a point transform: one input event in, one output event out. A `Pipeline`
composes several ops (and combinators) into a reusable graph, built once and run as many times
as you like on stored data or on a live event stream.

## `Input`: a named placeholder

`Input(name)` declares a source the graph will read from. It does not hold data itself; data is
bound later, when the pipeline is called:

```js
import { Input } from "@screamer-labs/screamer";

const x = Input("x");
```

## Composing ops into a graph

Passing an `Input` node (or any other node) through an op factory does not run the op. It
returns a symbolic node describing "apply this op here", and defers the actual computation
until the graph is compiled and called:

```js
import { RollingMean, Diff } from "@screamer-labs/screamer";

const x = Input("x");
const y = Diff(1)(RollingMean(3)(x));
```

`y` now describes a two-stage chain: a 3-period rolling mean, followed by a first difference.
Nothing has been computed yet.

## `Pipeline`: compile once, call many times

```js
import { Pipeline } from "@screamer-labs/screamer";

const pipeline = new Pipeline([x], [y]); // [declared inputs], [outputs]

const { values, index } = pipeline([1, 2, 3, 4, 5, 6]);

pipeline.dispose();
```

`new Pipeline(inputs, outputs)` compiles the graph once. Each call `pipeline(feeds)` binds data
to the declared inputs and runs the whole graph in a single pass, so a multi-op chain never
recomputes a subgraph shared by two outputs.

For a single-input pipeline, `feeds` can be a bare array or typed array. For multiple inputs, or
to supply an explicit index axis, pass an object keyed by input name:

```js
const a = Input("a");
const b = Input("b");
const p = new Pipeline([a, b], [...]);

p({
  a: { values: new Float64Array([1, 2, 3]), index: new Float64Array([0, 2, 4]) },
  b: new Float64Array([10, 20, 30]), // index defaults to 0, 1, 2, ...
});
```

A pipeline with one output returns a single `{ values, index }` object; a pipeline with several
outputs returns an array of them, one per declared output.

Dispose the pipeline with `pipeline.dispose()` when you are done with it. See
[Lifecycle](./lifecycle.md) for why you should not separately dispose the ops that went into
building the graph.

## Streaming with `pipeline.live()`

`pipeline.live()` returns an event-by-event driver over the same compiled graph. Push one event
at a time, then flush and drain the result:

```js
const driver = pipeline.live();
driver.push("x", /* index */ 0, /* value */ 1);
driver.push("x", 1, 2);
driver.push("x", 2, 3);
driver.flush();
const { values, index } = driver.result();
```

Streaming a pipeline this way reproduces its batch output exactly on the same data: `pipeline(data)`
and `pipeline.live()` fed the same events one by one always agree. `driver.result()` supports
single-output pipelines only, since draining reads every output's buffer at once.

## Combinators

Combinators are graph nodes that combine or reshape streams, rather than transforming one
stream's values. Each one takes node arguments and returns a node, so it composes into a graph
the same way an op does.

**`combineLatest(nodes, opts?)`**: as-of latest-value join across N node streams on possibly
different index axes. `emit: "when_all"` (the default) emits once every input has produced a
value; `emit: "on_any"` emits on any input event.

```js
import { combineLatest } from "@screamer-labs/screamer";

const joined = combineLatest([a, b]); // -> a wide 2-column stream, latest a & latest b
```

**`resample(node, opts)`**: causal windowed downsample. `every` buckets by a fixed index width;
`count` buckets by a fixed number of arrivals; `agg` picks the reducer (`"mean"`, `"last"`,
`"sum"`, `"ohlc"`, and others).

```js
import { resample } from "@screamer-labs/screamer";

const bars = resample(x, { every: 5, agg: "mean" });
// index 0..9 -> two bars labelled at 0 and 5, each the mean of its 5 points
```

**`select(node, columns)`**: pick one or more columns from a wide (multi-output) stream.

```js
import { select } from "@screamer-labs/screamer";

const lo = select(RollingMinMax(3)(x), 0); // the min column only
```

**`dropna(nodes, opts?)`**: drop events with NaN. `how: "any"` (the default) drops a row with
any NaN across its columns; `how: "all"` drops only rows that are all NaN. Useful right after
an op whose warmup produces NaN, to start a downstream computation only once real values start
flowing:

```js
import { dropna } from "@screamer-labs/screamer";

const warm = dropna(RollingMean(3)(x)); // skips the two NaN warmup events
```

**`filter(data, mask)`**: keep each `data` value whose aligned `mask` event is nonzero; zero or
NaN drops it.

```js
import { filter } from "@screamer-labs/screamer";

const gated = filter(x, mask);
```

**`delay(node, k)`**: re-stamp each event's index by `k` index units; values are unchanged.
Useful for lining up a signal with a lagged copy of itself before combining the two.

```js
import { delay } from "@screamer-labs/screamer";

const lagged = delay(x, 2);
```
