# screamer.js

screamer.js is the JavaScript build of [screamer](https://github.com/screamer-labs/screamer), a
causal streaming time-series operator library. It compiles the same C++ core that backs the
Python `screamer` package to WebAssembly, so it runs in Node.js and the browser with no native
addon and no separate `.wasm` file to fetch.

Every operator (rolling means, EMAs, RSI, drawdown, order-flow imbalance, and around two hundred
more) is the same C++ kernel the Python package calls. Results match Python to 1e-9; see
[Parity with Python](./parity.md).

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
or hand it a whole array and get the same result back in one pass:

```js
RollingMean(3)(new Float64Array([1, 2, 3, 4])); // Float64Array [NaN, NaN, 2, 3]
```

Start with [Getting started](./getting-started.md) for the async load step and your first
`Pipeline`.
