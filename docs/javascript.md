# JavaScript and the browser

`@screamer-labs/screamer` is the JavaScript package of screamer. It compiles the
same C++ operators to WebAssembly and runs in Node.js (18 or newer), through any
bundler, and directly in the browser. Use it to compute indicators next to a
live chart, in a web dashboard, or in a Node service, without a Python process.

## Install and first call

```bash
npm i @screamer-labs/screamer
```

The WebAssembly module loads asynchronously, so call `await ready()` once before
constructing an operator. An operator is called the same way as in Python:
construct it with its parameters, then call it on data.

```js
import { ready, RollingMean } from "@screamer-labs/screamer";

await ready();

const sma = RollingMean(3);
sma(1);    // NaN, window not yet full
sma(2);    // NaN
sma(3);    // 2

RollingMean(3)(new Float64Array([1, 2, 3, 4, 5]));
// Float64Array [ NaN, NaN, 2, 3, 4 ]
```

A number in gives a number out, so one value per event is the streaming form;
a `Float64Array` or a plain array in gives the same container back. Each
operator owns WebAssembly memory, so call `.dispose()` when you no longer need
it.

## In a web page

Without a build step, import the module from a CDN inside a module script:

```html
<script type="module">
  import { ready, RollingMean }
    from "https://cdn.jsdelivr.net/npm/@screamer-labs/screamer/dist/index.js";
  await ready();
  const sma = RollingMean(20);
</script>
```

The URL above loads the latest release; add `@<version>` after the package
name to pin one.

## Parity with Python

Both packages call one implementation of each operator: the C++ core, built as a
Python extension for `screamer` and as WebAssembly for `@screamer-labs/screamer`.
The JavaScript package has the same operators, parameters, warmup rules, and
causal behavior, and cross-language fixture tests check that its outputs match
the Python outputs to within 1e-9. The surface follows each language's idioms:
JavaScript option names are camelCase (`startPolicy` for `start_policy`), and
helper functions such as `position_test` become `positionTest`.

The function pages in this documentation therefore describe both packages. The
JavaScript-specific parts, such as module loading, async iterables, pipelines,
and memory management, are covered in the JavaScript guide.

## Live trades demo

The [live trades demo](https://screamer-labs.github.io/screamer/live-trades.html)
connects to a public exchange feed in your browser and runs screamer on every
trade as it arrives: OHLC bars, a low-lag price, a volume-weighted fair value,
buy and sell volume, and panels of momentum, volatility, order-flow, and cycle
indicators. Its source is a single HTML file,
[`js/examples/live-trades.html`](https://github.com/screamer-labs/screamer/blob/main/js/examples/live-trades.html).

## JavaScript reference

The [screamer.js documentation](https://screamer-labs.github.io/screamer/) has
the getting-started guide, calling conventions, pipelines, operator lifetime,
and the full API reference. The package is published on
[npm](https://www.npmjs.com/package/@screamer-labs/screamer).
