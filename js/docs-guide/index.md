screamer.js is the JavaScript build of [screamer](https://github.com/screamer-labs/screamer), a
library of causal time-series functions for finance and signal processing. It compiles the same
C++ core that backs the Python `screamer` package to WebAssembly, so it runs in Node.js and the
browser with no native addon and no separate `.wasm` file to fetch.

The package holds about 250 functions, each the same C++ kernel the Python package calls.
Results match Python to 1e-9; see [Parity with Python](./parity.md).

## What's in it

- Rolling, exponentially weighted, and expanding statistics: means, volatility, quantiles,
  z-scores, regressions.
- Technical indicators: trend, momentum, bands, volume.
- Filters and cycles: moving averages, Butterworth and Ehlers filters, dominant cycle.
- Market microstructure: trade signing, order-flow imbalance, price impact, VPIN.
- Bars from ticks: time, trade-count, and volume bars with `resample`.
- Backtest engines and reports, for one instrument or a portfolio.
- Significance tests for backtests: `positionTest`, `strategyTest`, and the randomizers they use.
- Pipelines: graphs of functions that run on stored arrays or on a live event stream.

## Install

```bash
npm i @screamer-labs/screamer
```

## 30 seconds

```js
import { ready, RollingMean } from "@screamer-labs/screamer";

await ready(); // load the WASM module once, before constructing any op

const sma = RollingMean(3);
sma(1); // NaN, window not yet full
sma(2); // NaN
sma(3); // 2
sma(4); // 3

sma.dispose(); // free the op's WASM-side state when done
```

`RollingMean(3)` returns a stateful, callable op. Call it with one number at a time (as above),
or hand it a whole array in one call:

```js
RollingMean(3)(new Float64Array([1, 2, 3, 4])); // Float64Array [NaN, NaN, 2, 3]
```

Start with [Getting started](./getting-started.md) for the async load step and your first
`Pipeline`, or watch the operators run on a live trade feed in the
[Live demo](./live-demo.md).

## Built-in documentation

The documentation of every function ships with the package, in JavaScript names, for people and
coding agents:

```bash
npx screamer-docs search "order flow imbalance"
npx screamer-docs show RollingRSI
```

See [Built-in documentation](./documentation.md) for the commands and the
`@screamer-labs/screamer/docs` module.
