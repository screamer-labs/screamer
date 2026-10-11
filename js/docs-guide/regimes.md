# Calling conventions

An op factory such as `RollingMean(3)` or `RollingPoly2(50, 1)` returns a callable. That
callable dispatches on the type of its argument, and every regime below runs the same causal,
no-lookahead computation. A call with one number continues the op's running series; every other
call is an independent run that starts clean, and a stream runs on copies of the op that it
owns.

## Streaming: `op(number)` -> `number`

Pass one number, get one number back. This is the regime for driving an op event by event, such
as inside a `for` loop over incoming ticks:

```js
const sma = RollingMean(3);
sma(1); // NaN
sma(2); // NaN
sma(3); // 2
sma(4); // 3
```

## Batch: `op(Float64Array | number[])` -> same container type

Pass a whole series, get the whole output series back in one call, computed as if you had
called the op once per element in order. The output container matches the input: a
`Float64Array` in gives a `Float64Array` out, a plain `number[]` in gives a `number[]` out.

```js
RollingMean(3)(new Float64Array([1, 2, 3, 4])); // Float64Array [NaN, NaN, 2, 3]
RollingMean(3)([1, 2, 3, 4]);                   // [NaN, NaN, 2, 3]
```

A batch call starts from a clean warmup regardless of what was called on the op before, and
it leaves the op's running series reset, so the next `op(number)` starts a new warmup.

## Sync iterable: `op(iterable)` -> iterator

Pass any iterable of numbers (not a `Float64Array` or `Array`, which are handled by the batch
regime above), get an iterator back that yields one output per input value. It reads nothing
from the source before the first `next()`:

```js
function* ticks() {
  yield 1; yield 2; yield 3; yield 4;
}

for (const y of RollingMean(3)(ticks())) {
  console.log(y); // NaN, NaN, 2, 3
}
```

## Async iterable: `op(asyncIterable)` -> async iterator

Pass an async iterable, get an async iterator back, useful for streaming data off a
`ReadableStream`, a websocket, or any other async source:

```js
async function* ticks() {
  for (const v of [1, 2, 3, 4]) yield v;
}

for await (const y of RollingMean(3)(ticks())) {
  console.log(y); // NaN, NaN, 2, 3
}
```

## Streams are independent runs

A stream, sync or async, starts clean and runs on copies of the op that it owns. Two streams
of one op never share state, whether they run one after the other or interleaved, and each
gives the result of its own batch call. Opening a stream resets the op's running series, like
a batch call, and later `op(number)` calls do not touch a stream that is already open:

```js
const mean = RollingMean(3);
const a = mean([1, 2, 3, 4, 5, 6].values());
const b = mean([100, 200, 300, 400, 500, 600].values());
for (let i = 0; i < 6; i++) console.log(a.next().value, b.next().value);
// NaN NaN, NaN NaN, 2 200, 3 300, 4 400, 5 500
```

The [streaming guide](https://screamer.readthedocs.io/en/latest/streaming.html) of the Python
documentation describes the rest in detail: laziness, several sources, asset events, and
errors.

## Multi-input ops

An op that takes several inputs (`Add()`, for example) extends the same regimes column-wise:
call it with N numbers for one streaming event, or N same-length arrays for a columnar batch.

```js
Add()(1, 2);                                              // 3
Add()(new Float64Array([1, 2, 3]), new Float64Array([10, 20, 30])); // Float64Array [11, 22, 33]
Add()([1, 2, 3], [10, 20, 30]);                           // [11, 22, 33]
Add()([1, 2, 3], new Float64Array([10, 20, 30]));         // Float64Array [11, 22, 33]
Add()([[1, 10], [2, 20]]);                                // [11, 22], one result per event
```

The container follows the inputs: all `number[]` inputs give a `number[]`, and any
`Float64Array` or `NdArray` input gives a `Float64Array` or `NdArray`. An array of events gives
an array with one result per event. Mismatched array lengths across inputs throw a `RangeError`
rather than truncating or NaN-filling.

A stream takes one iterable per input, or one iterable whose events hold every input, such as
`[high, low, close]` arrays. Each event reads one item from every source, and the stream ends
with the shortest source. All inputs are sync iterables or all are async iterables; mixing the
two, or mixing an iterable with an array, throws a `TypeError`.

```js
function* prices() { yield 1; yield 2; yield 3; }
function* volumes() { yield 10; yield 20; yield 30; }

[...Mul()(prices(), volumes())];          // [10, 40, 90]
[...Mul()([[1, 10], [2, 20]].values())]; // [10, 40]
```

## Assets: `[T, K]` `NdArray` input

An `NdArray` of shape `[T, K]` holds `K` series (assets) side by side, with time on the first
axis. A one-input op runs on each column with its own state, and a multi-input op broadcasts its
inputs over assets, so a `[T]` series pairs with every column of a `[T, K]` one:

```js
const X = { data: new Float64Array([1, 10, 2, 20, 3, 30]), shape: [3, 2] };
RollingMean(2)(X);                                  // NdArray [3, 2]: [NaN, NaN, 1.5, 15, 2.5, 25]
Add()(X, new Float64Array([100, 200, 300]));        // NdArray [3, 2]: [101, 110, 202, 220, 303, 330]
```

An iterable of `[K]` rows gives the same rows one event at a time, one op state per asset. A
multi-input stream broadcasts its events like a batch call, so a source of numbers pairs with a
source of `[K]` rows, and one source of `[K, N]` events holds the N inputs of an N-input op.
Shapes that do not broadcast throw a `RangeError`. `reshape(x, layout)` names the axes of an
`NdArray` with the letters `T` (time), `A` (asset), and `F` (field), as in Python.

## Multi-output ops: `NdArray` and `toNested()`

An op that produces more than one value per event (`RollingMinMax`, for example) returns a
plain object per event or an `NdArray` for a batch, rather than a wider array-of-arrays:

```ts
interface NdArray {
  data: Float64Array; // row-major, shape[0] * shape[1] entries
  shape: number[];    // [rows, columns]
}
```

```js
import { RollingMinMax, toNested } from "@screamer-labs/screamer";

const out = RollingMinMax(3)(new Float64Array([1, 2, 3, 4, 5]));
// out.data is a flat Float64Array, out.shape is [5, 2]

toNested(out); // [[1,1], [1,2], [1,3], [2,4], [3,5]] -- expanding window before it fills
```

`toNested()` unpacks an `NdArray` into an array of rows (or a flat array, for a single-column
result). Reach for the flat `data`/`shape` form when you want to avoid the allocation of
per-row arrays; reach for `toNested()` when you want ordinary nested arrays to iterate over.

## Multi-asset reducers: reducing a variable number of series

Most ops read a fixed number of values per event, decided when the op is constructed. A
reducer instead folds a variable number of assets into one event, and reads that count from
the data on every call. `PortfolioReport` is the one such op: it reduces the output of a
backtest engine, run over any number of assets, into portfolio-level report columns.

An event is an `(assets, 4)` block, and a batch is an `(events, assets, 4)` one:

```js
import { PortfolioReport } from "@screamer-labs/screamer";

const report = PortfolioReport();

// One event: three assets, each contributing [equity, pnl, position, cost].
report([
  [100.0, 1.0, 2.0, 0.02],
  [250.0, -0.5, -1.0, 0.01],
  [ 80.0, 0.0, 0.0, 0.0 ],
]); // [drawdown, cumCost, turnover, trades, maxDrawdown, sharpe]

// A whole run at once, as an NdArray of shape [events, assets, 4].
report({ data: engineOutput, shape: [events, assets, 4] }); // NdArray, shape [events, 6]
```

An array of events (nested arrays, or `[assets, 4]` `NdArray` events) gives an array of
`Float64Array` result rows, one per event.

The nested form above and a flat `Float64Array` of `assets * 4` values are both accepted for a
single event, as is an iterable of events for streaming. The asset count is fixed by the first
event after a `reset()`; changing it later throws, rather than silently redefining what the
portfolio is.
