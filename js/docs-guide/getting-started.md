# Getting started

## Install

```bash
npm i @screamer-labs/screamer
```

The package ships one self-contained WASM module. There is no separate `.wasm` asset to fetch
and no build step on the consumer side.

## Load the module once

screamer.js compiles to WebAssembly, and WebAssembly modules instantiate asynchronously
(compiling the binary and linking its imports both happen off the main thread). Call `await
ready()` once, before constructing any op:

```js
import { ready, RollingMean } from "@screamer-labs/screamer";

await ready();
```

Every op factory reads the loaded module internally, so calling one before `ready()` resolves
throws. A repeated `ready()` call is cheap: it awaits the same cached load rather than
re-instantiating the module.

## Your first operator

An op factory like `RollingMean(windowSize)` returns a stateful, callable op:

```js
const sma = RollingMean(3);

sma(1); // NaN, window not yet full
sma(2); // NaN
sma(3); // 2
sma(4); // 3
```

Each call is one causal step: it consumes the new value and the op's own history, and never
looks ahead. The op owns WASM-side memory, so call `sma.dispose()` when you are done with it
rather than waiting on garbage collection. See [Calling conventions](./regimes.md) for the other
three ways to call an op, and [Lifecycle](./lifecycle.md) for disposal in full.

## Your first pipeline

A single op is a point transform. A `Pipeline` composes several into a reusable graph, built
once from `Input` placeholders and run on stored data as many times as you like:

```js
import { ready, Input, Pipeline, RollingMean, Diff } from "@screamer-labs/screamer";

await ready();

const x = Input("x");
const y = Diff(1)(RollingMean(3)(x));
const pipeline = new Pipeline([x], [y]);

const { values, index } = pipeline([1, 2, 3, 4, 5, 6]);

pipeline.dispose();
```

`Diff(1)(RollingMean(3)(x))` does not run anything yet: passing an `Input` node through an op
factory returns a symbolic node describing the computation. `new Pipeline([x], [y])` compiles
that graph, and `pipeline(data)` binds `x` to `data` and runs the whole chain in one pass. See
[Pipelines and graphs](./pipeline.md) for multi-input graphs, streaming with `pipeline.live()`,
and the combinators (`combineLatest`, `resample`, and friends).
