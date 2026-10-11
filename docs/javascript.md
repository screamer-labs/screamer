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
causal behavior. Results agree to within rounding error but are not always
identical bit for bit, because the native and WebAssembly builds use different
math libraries and floating-point code generation; cross-language fixture tests
check the outputs against Python to within 1e-9. Seeded randomization, such as
the copies drawn by `positionTest`, is identical in both. The surface follows
each language's idioms: names are camelCase (`startPolicy` for `start_policy`),
and helper functions such as `position_test` become `positionTest`.

The function pages in this documentation describe each operator's behavior and
parameters, which are the same in both packages, but their examples are in
Python. A JavaScript constructor takes the parameters positionally, in the order
the page lists them, so `EwMean(span=20)` becomes `EwMean(undefined, 20)`;
passing `undefined` keeps a parameter's default. Helpers that take options,
such as `positionTest`, take them as an object. A few Python helpers have no
JavaScript counterpart yet: `backtest_report` (use the `BacktestReport` and
`BacktestStats` operators), `forecast_pairs`, `split`, and the pandas
conversions. The JavaScript-specific parts, such as module loading, async
iterables, pipelines, and memory management, are covered in the JavaScript
guide.

### Parity gaps

These parts of the shape contract work in one package but not yet the other.
Each is tracked to be closed.

* `combineLatest` in JavaScript has no `layout` argument.
* The microstructure synonyms `SignedVolume`, `RollingKyleLambda`,
  `EwKyleLambda`, `RollingOrderImbalance`, and `QueueImbalance` are Python
  only; in JavaScript use `Mul`, `RollingBeta` and `EwBeta` (with the inputs
  as return, then signed flow), `RollingSum`, and `OFI` directly.
* In Python, a live session's `push` takes numbers only, not asset events.

## Backtesting and significance tests

The backtest engines and the significance tests are in the JavaScript package
too. An engine with more than one output returns an `NdArray`, an object with a
row-major `data` array (`Float64Array`) and a `shape`. The examples read the
repository's sample data with Node's `fs`, so run them from the repository
root.

A single-instrument backtest of a trend rule on daily BTC bars, with a 10 bps
fee and 5 bps slippage, then a test of its timing against 999 randomly shifted
copies:

```js
import { readFileSync } from "node:fs";
import {
  ready, EwMean, Sub, Sign, FillNa, BacktestOHLCTarget, BacktestReport, positionTest,
} from "@screamer-labs/screamer";

await ready();

const rows = readFileSync("docs/notebooks/data/binance_daily_20.csv", "utf8")
  .trim().split("\n").slice(1).map((line) => line.split(","))
  .filter((f) => f[1] === "BTC");
const [open, high, low, close] = [2, 3, 4, 5].map(
  (i) => Float64Array.from(rows, (f) => Number(f[i])));

// +1 when the 20-day EwMean is above the 100-day one, -1 below
const spread = Sub()(EwMean(undefined, 20)(close), EwMean(undefined, 100)(close));
const signal = Sign()(FillNa(0)(spread));

// BacktestOHLCTarget(takerFee, tickSize, minPosition, maxPosition,
//                    multiplier, feePerContract, slippage, ...)
const engine = BacktestOHLCTarget(1e-3, 0, -Infinity, Infinity, 1, 0, 5e-4);
const out = engine(signal, open, high, low, close);   // shape [T, 4]

const T = out.shape[0];
const column = (k) => Float64Array.from({ length: T }, (_, t) => out.data[t * 4 + k]);
const report = BacktestReport(365)(column(0), column(1), column(2), column(3));
console.log("pnl", column(0)[T - 1].toFixed(0),
            "Sharpe", report.data[(T - 1) * 6 + 5].toFixed(2));   // pnl 56362 Sharpe 0.40

const test = positionTest(engine, signal, [open, high, low, close],
                          { n: 999, periodsPerYear: 365 });
console.log("p-value", test.pValue);                                // p-value 0.151
```

The engine output columns are `[equity, pnl, position, cost]`, and the
`BacktestReport` columns are `[drawdown, cum_cost, turnover, trades,
max_drawdown, sharpe]`, so the last row of each holds the summary. The numbers
match the [first strategy recipe](recipes/first-strategy.md), which runs the
same backtest in Python.

A portfolio engine takes `(T, K)` inputs, given as an array of rows or as an
`NdArray`, and a one-input operator such as `LogReturn` runs on each column of
a `(T, K)` `NdArray`. This example ranks five coins on 30-day momentum, turns
the ranks into dollar-neutral weights with `CrossNormalize`, and rebalances weekly: a
`NaN` weight keeps the asset's position. `PortfolioReport` reduces the
`(T, K, 4)` output to the book's report, and `positionTest` on a portfolio
engine tests the book as a whole.

```js
import { readFileSync } from "node:fs";
import {
  ready, LogReturn, CrossRank, CrossNormalize, BacktestPortfolioOHLCTarget,
  PortfolioReport, positionTest,
} from "@screamer-labs/screamer";

await ready();

const coins = ["BTC", "ETH", "SOL", "XRP", "BNB"];
const lines = readFileSync("docs/notebooks/data/binance_daily_20.csv", "utf8")
  .trim().split("\n").slice(1).map((line) => line.split(","));
const columns = (field) => coins.map((s) =>
  Float64Array.from(lines.filter((f) => f[1] === s), (f) => Number(f[field])));
const toRows = (cols) => Array.from(cols[0], (_, t) => cols.map((c) => c[t]));
const [open, high, low, close] = [2, 3, 4, 5].map((i) => toRows(columns(i)));

const K = coins.length;
const closes = { data: Float64Array.from(close.flat()), shape: [close.length, K] };
const momentum = LogReturn(30)(closes);                         // NdArray [T, K]
const weights = CrossNormalize(1, 0)(CrossRank()(momentum));    // NdArray [T, K]
weights.data.forEach((_, i) => { if (Math.floor(i / K) % 7 !== 0) weights.data[i] = NaN; });

// BacktestPortfolioOHLCTarget(fee, slippage, ...): weights of equity, next-open fills
const engine = BacktestPortfolioOHLCTarget(1e-3, 5e-4);
const out = engine(weights, open, high, low, close);           // shape [T, K, 4]
const report = PortfolioReport()(out);                          // shape [T, 6]

const T = out.shape[0];
let equity = 1;                                                 // initialCapital
for (let k = 0; k < K; k++) equity += out.data[((T - 1) * K + k) * 4];
console.log("equity", equity.toFixed(3),
            "max drawdown", report.data[(T - 1) * 6 + 4].toFixed(3)); // equity 1.064 max drawdown -0.454

const test = positionTest(engine, weights, [open, high, low, close], { n: 999 });
console.log("p-value", test.pValue);                            // p-value 0.174
```

`strategyTest`, `probabilisticSharpe`, and `deflatedSharpe` are available under
the same camelCase names.

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
