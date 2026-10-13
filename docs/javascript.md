# JavaScript and the browser

`@screamer-labs/screamer` is the JavaScript package of screamer. It compiles the
same C++ operators to WebAssembly and runs in Node.js (18 or newer), through any
bundler, and directly in the browser. Use it to compute indicators next to a
live chart, in a web dashboard, or in a Node service, without a Python process.

The WebAssembly module uses native WebAssembly exceptions, so it needs Chrome 95,
Firefox 100, Safari 15.2 or Node 17 or later. The package itself supports
Node 18 or newer.

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
Python. A JavaScript constructor takes the parameters by name in one options
object, so `EwMean(span=20)` becomes `EwMean({ span: 20 })`. The names are
camelCase, an unknown name raises `TypeError`, and each operator has a matching
TypeScript type (`EwMeanOptions`). The options object must be the only argument.
A constructor also takes the parameters positionally, in the order the page
lists them: `EwMean(undefined, 20)`. In both forms, `undefined` keeps a
parameter's default. `null` means "not given" only for the parameters that
Python marks Optional and takes `None` for, such as the `com`, `span`,
`halflife` and `alpha` of the exponentially weighted operators; for any other
parameter, `null` raises `TypeError`, as `None` does in Python. A number
parameter takes only a number: a string, a boolean, an array or an object
raises `TypeError`. Leaving out a parameter that has no default, or passing
more arguments than the operator has parameters, also raises `TypeError`. An
operator without parameters, such as `Abs`, takes no argument or an empty
object. Helpers that take options, such as `positionTest`, take
them as an object. The pandas
conversions have no JavaScript counterpart. The
JavaScript-specific parts, such as module loading, async iterables, pipelines,
and memory management, are covered in the JavaScript guide.

### The six rules in JavaScript

The [six rules](polymorphic_api.md#the-six-rules) hold in JavaScript, with an
`NdArray` (`{ data, shape }`) in the place of a NumPy array:

1. **Shape `[T, ...A, F?]`.** On an `NdArray` of rank 3 or more, a trailing
   axis of size 1 is the single input field: `RollingMean(3)` on shape
   `[T, K, 1]` returns shape `[T, K]`, and shape `[T, 1]` stays `[T, 1]`.
2. **The container never changes the numbers.** `[...RollingMean(3)(rows.values())]`,
   with `rows` an array of `[K]` rows, gives the rows of the batch result.
3. **One meaning per container.** A number is one step of the op's running
   series, a `Float64Array`, a `number[]` or an `NdArray` is a batch, and an
   iterable or async iterable is a stream of events. Each stream is an
   independent run with its own state. An array of events, such as
   `[[x0, y0], [x1, y1]]` or `[Float64Array(K), Float64Array(K)]`, is a
   sequence of events and gives an array with one result per event, as a
   Python list does. All `number[]` inputs give a `number[]` (an array of
   `number[]` for a multi-output op), as Python lists give a list, and a call
   that mixes `number[]` with `Float64Array` or `NdArray` inputs returns an
   array. In a `Pipeline` feed or a live push,
   `null` is a missing value and reads as `NaN`, as Python's `None` does.
4. **Separate arguments, or one packed array, with assets broadcast.**
   `RollingCorr(20)(X, y)` with `X` of shape `[T, K]` and `y` a `Float64Array`
   of length `T` gives `[T, K]`, and so does one packed `[T, K, 2]` `NdArray`.
5. **Ambiguity throws.** Inputs of different lengths and asset shapes that do
   not broadcast throw `RangeError`, and a wrong kind of input throws
   `TypeError`.
6. **A change to the result of working code needs an explicit decision** and
   an entry under "Changed" in the release notes.

[Streams in depth](streaming.md) describes stream state, laziness, several
sources, asset events, and errors.

### Parity gaps

These parts work differently in the two packages, or in one package only. Each
is tracked to be closed.

* Error classes differ: Python raises `ValueError` for inputs or events whose
  shapes do not match, where JavaScript throws `RangeError` (lengths, event
  shapes, asset shapes that do not broadcast). A wrong kind of input throws
  `TypeError` in both.
* A nested-array event for an `N`-input op lists the `N` fields first; use an
  `NdArray` of shape `[K, N]` to list the assets first, as a `(K, N)` NumPy
  event does.

### Intentional differences

These differences follow each language's own convention. They are not gaps and
are not tracked to be closed.

* An indexed stream is marked, never inferred from its items. Python marks it
  with a tuple: `(values, index)` for an array, `(value, index)` for an event
  of an iterator. JavaScript has no tuple, so it marks it with an object:
  `{values, index}` for an array (as a `Pipeline` feed, `combineLatest` input
  or `split` output) and `{value, index}` for an event of an iterator (as a
  `combineLatest` source). A plain Array, a typed array or an `NdArray` is
  always values in both languages: `[[1, 2], [3, 4]]` is two rows of two
  values, as the Python list `[[1, 2], [3, 4]]` is.
* The microstructure operators `SignedVolume`, `RollingKyleLambda`,
  `EwKyleLambda`, `RollingOrderImbalance` and `QueueImbalance` also take their
  inputs as keywords in Python (`SignedVolume()(sign=s, volume=v)`).
  JavaScript has no keyword arguments, so pass the inputs in order.
* Awaiting a second `next()` on an async stream while the first is still
  running raises `RuntimeError` in Python. JavaScript queues the calls and
  answers them in order, as an async generator does.
* An error during a stream: Python raises from that `next()`, keeps
  the stream open, and reads the next event on the following call, closing the
  sources only when the stream is garbage collected. JavaScript throws, closes
  every source, and ends the stream.

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
  ready, EwMean, Sub, Sign, FillNa, BacktestOHLCTarget, backtestReport, positionTest,
} from "@screamer-labs/screamer";

await ready();

const rows = readFileSync("docs/notebooks/data/binance_daily_20.csv", "utf8")
  .trim().split("\n").slice(1).map((line) => line.split(","))
  .filter((f) => f[1] === "BTC");
const [open, high, low, close] = [2, 3, 4, 5].map(
  (i) => Float64Array.from(rows, (f) => Number(f[i])));

// +1 when the 20-day EwMean is above the 100-day one, -1 below
const spread = Sub()(EwMean({ span: 20 })(close), EwMean({ span: 100 })(close));
const signal = Sign()(FillNa(0)(spread));

const engine = BacktestOHLCTarget({ takerFee: 1e-3, slippage: 5e-4 });
const out = engine(signal, open, high, low, close);   // shape [T, 4]

const { summary } = backtestReport(out, { periodsPerYear: 365 });
console.log("pnl", summary.total_pnl.toFixed(0),
            "Sharpe", summary.sharpe.toFixed(2));                 // pnl 56362 Sharpe 0.40

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
`NdArray`. Give it an `NdArray` and it returns an `NdArray`; give it only arrays
of rows and it returns an array of result rows. A one-input operator such as `LogReturn` runs on each column of
a `(T, K)` `NdArray`. This example ranks five coins on 30-day momentum, turns
the ranks into dollar-neutral weights with `CrossNormalize`, and rebalances weekly: a
`NaN` weight keeps the asset's position. `PortfolioReport` reduces the
`(T, K, 4)` output to the book's report; with the fill prices as a second
argument its turnover is the traded notional rather than units summed across
coins. `positionTest` on a portfolio engine tests the book as a whole.

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
const report = PortfolioReport()(out, open);                    // shape [T, 6]

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
