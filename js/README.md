# @screamer-labs/screamer

WASM build of [screamer](https://github.com/screamer-labs/screamer), a high-performance
causal streaming time-series operator library. It runs the same C++ core as the Python
`screamer` package, compiled to WebAssembly, in Node.js and the browser.

## Install

```bash
npm i @screamer-labs/screamer
```

The package ships a single self-contained WASM module; there is no separate `.wasm`
asset to fetch and no build step on the consumer side.

## The WASM module loads asynchronously

Call `await ready()` once, before constructing any op. Every factory reads the loaded
module, so using one beforehand throws.

```js
import { ready, RollingMean } from "@screamer-labs/screamer";

await ready();
const op = RollingMean(3);
```

## Calling an op: four input regimes

An op factory like `RollingMean(3)` returns a callable. That callable dispatches on its
argument's type and preserves the container shape of its input: a scalar in gives a
scalar out, a typed array in gives a typed array out, and so on.

```js
import { ready, RollingMean } from "@screamer-labs/screamer";

await ready();

// number -> number, one event at a time (the streaming regime).
const live = RollingMean(3);
live(1); // NaN, window not yet full
live(2); // NaN
live(3); // 2

// Float64Array -> Float64Array, container-preserving.
const fa = RollingMean(3)(new Float64Array([1, 2, 3, 4, 5]));
// Float64Array [ NaN, NaN, 2, 3, 4 ]

// number[] -> number[], container-preserving.
const arr = RollingMean(3)([1, 2, 3, 4, 5]);
// [ NaN, NaN, 2, 3, 4 ]

// async iterable -> async iterable, yielding one output per input event.
async function* events() {
  for (const v of [1, 2, 3, 4, 5]) yield v;
}
const out = [];
for await (const y of RollingMean(3)(events())) out.push(y);
// [ NaN, NaN, 2, 3, 4 ]
```

Each op instance is stateful and owns WASM-side memory. Call `.dispose()` when done
with it (`live.dispose()` above), rather than waiting on garbage collection.

## Composing a pipeline

`Input` declares a named placeholder; passing it through op factories builds a
symbolic graph without running anything. `Pipeline` compiles that graph once and
returns a reusable function you call on stored data.

```js
import { ready, Input, Pipeline, RollingMean, Diff } from "@screamer-labs/screamer";

await ready();

const x = Input("x");
const y = Diff(1)(RollingMean(3)(x));
const pipeline = new Pipeline([x], [y]);

const { values, index } = pipeline([1, 2, 3, 4, 5, 6]);

pipeline.dispose();
```

`pipeline(feeds)` binds the declared inputs to data and runs the compiled graph in one
pass, so a multi-op chain does not recompute shared subgraphs. `pipeline.live()`
returns an event-by-event driver for streaming input; see the type definitions for its
`push`/`advance`/`flush`/`result` methods.

## Parity with Python

This package is the JS/WASM build of the Python [`screamer`](https://pypi.org/project/screamer/)
package: same operators, same causal semantics, same numerics. Its outputs are
verified against the Python package's outputs, and batch and streaming calls on the
same data give identical results.

## Live demo

Watch the smoothing operators run on a live exchange trade feed in the browser:
[screamer-labs.github.io/screamer/live-trades.html](https://screamer-labs.github.io/screamer/live-trades.html).
The self-contained source is [`examples/live-trades.html`](examples/live-trades.html).

## Learn more

The JavaScript reference and guide are at
[screamer-labs.github.io/screamer](https://screamer-labs.github.io/screamer/). Full documentation,
the function reference, and example notebooks are at
[screamer.readthedocs.io](https://screamer.readthedocs.io/en/latest/). Source and
issues live at [github.com/screamer-labs/screamer](https://github.com/screamer-labs/screamer).
