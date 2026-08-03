# Calling conventions

An op factory such as `RollingMean(3)` or `RollingPoly2(50, 1)` returns a callable. That
callable dispatches on the type of its argument, and every regime below shares the same op
state and the same causal, no-lookahead computation. Only the calling convention changes.

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

A batch call resets the op's internal state first, so it always starts from a clean warmup
regardless of what was called on the op before.

## Sync iterable: `op(iterable)` -> generator

Pass any iterable of numbers (not a `Float64Array` or `Array`, which are handled by the batch
regime above), get a generator back that yields one output per input value, lazily:

```js
function* ticks() {
  yield 1; yield 2; yield 3; yield 4;
}

for (const y of RollingMean(3)(ticks())) {
  console.log(y); // NaN, NaN, 2, 3
}
```

## Async iterable: `op(asyncIterable)` -> async generator

Pass an async iterable, get an async generator back, useful for streaming data off a
`ReadableStream`, a websocket, or any other async source:

```js
async function* ticks() {
  for (const v of [1, 2, 3, 4]) yield v;
}

for await (const y of RollingMean(3)(ticks())) {
  console.log(y); // NaN, NaN, 2, 3
}
```

## Multi-input ops

An op that takes several inputs (`Add()`, for example) extends the same regimes column-wise:
call it with N numbers for one streaming event, or N same-length arrays for a columnar batch.

```js
Add()(1, 2);                                              // 3
Add()(new Float64Array([1, 2, 3]), new Float64Array([10, 20, 30])); // Float64Array [11, 22, 33]
```

Mismatched array lengths across inputs throw a `TypeError` rather than truncating or
NaN-filling.

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

## Multi-group ops: reducing a variable number of series

Most ops read a fixed number of values per event, decided when the op is constructed. A
reducer instead folds a variable number of groups into one event, and reads that count from
the data on every call. `PortfolioReport` is the one such op: it reduces the output of a
backtest engine, run over any number of assets, into portfolio-level report columns.

An event is a `(groups, 4)` block, and a batch is a `(events, groups, 4)` one:

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

The nested form above and a flat `Float64Array` of `groups * 4` values are both accepted for a
single event, as is an iterable of events for streaming. The group count is fixed by the first
event after a `reset()`; changing it later throws, rather than silently redefining what the
portfolio is.
