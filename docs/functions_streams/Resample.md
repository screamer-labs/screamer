---
name: Resample
title: Resample
kind: class
short: Causal windowed downsample of a 1-D or multi-column value stream.
topics:
- streams
covers:
- resample
---

# `Resample`

Causal windowed downsampling. Group a stream into fixed index-interval buckets
(`freq=`) or fixed event-count buckets (`count=`), reduce each bucket with a
per-bar aggregation, and return a `(values, index)` tuple. A bucket emits only once a
later index proves it complete; the trailing partial bucket emits at end of input.
Usable eagerly (raw arrays or `(values, index)` tuples) and inside a `Pipeline`.

Feeding a lazy iterator of `(value, index)` pairs returns a lazy iterator of bar events; feeding arrays or `(values, index)` tuples returns the batch result.

The `agg` parameter accepts two forms:

**String shorthand**: one of `first`, `last`, `min`, `max`, `sum`, `count`,
`mean`, `ohlc`, `ohlcv`, `ohlcv2`, `ohlc_bars`, `ohlcv_bars`. `ohlc` returns
four columns (`open`, `high`, `low`, `close`). The multi-column bar aggs
require a 2-D input; see below. All multi-column aggs work identically in
eager, `Pipeline` graph, and lazy iterator regimes.

**Any `EvalOp` functor**, e.g. `ExpandingSkew()`. The functor is `reset()` at
each bar boundary and fed every in-bar sample; its last output before the close
is emitted as the bar value. All screamer functors are valid reducers.

**Several reducers at once**: run one `Resample` per statistic over the same
bucketing and align the results with `CombineLatest`. Because every `Resample`
shares the same `freq=` (or `count=`) and clock, the bars line up exactly and
cannot drift, e.g.
`CombineLatest()(Resample(freq=5, agg="first")(price, t), Resample(freq=5, agg="sum")(vol, t))`.
Inside a `Pipeline`, place each per-stat `Resample` node on its upstream expression
and combine them the same way; per-tick transforms live in the expression, e.g.
`Resample(freq=5, agg=ExpandingSum())(PosPart()(vol))`.

## Bucketing: `freq=` vs `count=`

Exactly one of `freq=` or `count=` sets how bars are bounded, and they answer
different questions.

- `freq=W` buckets along the **index**. Bar `n` is the half-open interval
  `[origin + n*W, origin + (n+1)*W)`, so the index values decide membership. Bars
  have equal width on the index but a variable number of ticks; a tick exactly on a
  boundary starts the later bar. Boundaries are anchored at `origin` (default `0`,
  i.e. multiples of `W`), **not** at the first tick. Set `origin=` to shift the
  grid. Internal empty intervals are real and controlled by `fill=`.
- `count=N` buckets by **arrival order**. A bar closes every `N` events and never
  consults the index values to place boundaries. Bars have an equal number of ticks
  but a variable width on the index, and one bar can straddle an arbitrary index gap
  without noticing it.

The `index` argument is **optional in both modes**. `count=` does not need it to
find boundaries; `freq=` uses it as the timeline being bucketed. If omitted, row
position (`0, 1, 2, ...`) is used as the index.

Bar **labels** depend on the mode and on `label=`:

- `freq=`: the bar's **grid edge**, `origin + n*W` for `label="left"` (default)
  or `origin + (n+1)*W` for `label="right"`. This is the interval boundary itself,
  which need not equal any actual tick's index (and a right label can sit past the
  last tick).
- `count=`: an **actual tick index**, the **first** tick of the bar for
  `label="left"` or the **last** for `label="right"`.

Concretely, eight ticks at index `[0, 1, 2, 10, 11, 20, 21, 22]`:

- `freq=10` -> bars `{0,1,2} {10,11} {20,21,22}` (counts 3, 2, 3), labels
  `[0, 10, 20]` (grid edges).
- `count=3`  -> bars `{0,1,2} {10,11,20} {21,22}` (counts 3, 3, 2), labels
  `[0, 10, 21]` (first tick of each bar). The middle bar straddles the `11 -> 20`
  gap because `count=` measures rows, not index distance.

## Empty buckets: `fill=`

`fill=` controls what happens when a `freq=` bar contains no samples (an index
interval with no ticks). It applies to eager arrays, `(values, index)` tuples, and graphs alike;
it is not `Pipeline`-only.

- `"skip"` (default): emit no row for an empty bucket (the legacy behavior).
- `"nan"`: emit an all-NaN row at the empty bucket's label.
- `"carry"`: repeat the previous emitted row's values verbatim.

Only **internal** empty buckets (gaps between two events) are filled by `Resample`
itself. Trailing empty buckets after the last event are not synthesized here; that
needs a clock, via either [`pipe.live().advance(now)`](#pipeline-live) or a clock input
wired into a `Pipeline`. The [`pipe.live()` reference](#pipeline-live) has a worked example.

`fill=` is meaningful only under `freq=`. With `count=`, a bar is defined by
holding `N` events, so empty bars cannot exist by construction and `fill=` has no
effect.

### `fill=` edge cases

- **Leading buckets** (before the first event) are never synthesized on the eager
  path: output starts at the bucket containing the first tick. A tick at index 47
  with `freq=10` starts at label 40, not 0. Leading empty buckets arise only from a
  clock (a `Pipeline` clock input or [`advance()`](#pipeline-live)) that advances time before
  the first data event; there `"nan"` emits NaN rows and `"carry"` **skips** them,
  since there is no previous row to repeat.
- **`"carry"` is verbatim and uniform** across every column: it repeats the previous
  bar's whole row, including count-like or sum-like columns. So a genuinely empty bar
  carries the previous bar's count, not 0. Use `"nan"` where that is wrong for your
  columns; a per-column fill policy is out of scope in v1.
- **`label="right"`** composes with `fill=` unchanged: a filled bucket is labelled at
  its right grid edge, like any other bar.
- **Functor reducers** are unaffected. A filled empty bucket emits a synthetic row
  (a NaN row, or the carried row) without feeding or resetting the reducer; the
  reducer is `reset()` only at the next real bar boundary. An expanding reducer
  therefore starts each real bar clean and never accumulates across a filled gap.

## Output format and column order

Every `Resample` call returns a `(values, index)` tuple. Unpack it as
`values, index = Resample(...)(data, idx)`. Multi-column aggregations (`ohlc`,
`ohlcv`, `ohlcv2`, `ohlc_bars`, `ohlcv_bars`) return a 2-D values array; columns
are positional in documented order. Use
`to_pandas(values, index, columns=["open","high","low","close"])` to attach names
for display. See [OHLC column order](Stream.md#ohlc-column-order) for the full
column listing.

## `ohlcv` and `ohlcv2` (two-column input)

Both require `values` to be a `(T, 2)` array: column 0 is price, column 1 is
volume (unsigned for `ohlcv`, signed for `ohlcv2`). Both work in eager, graph,
and lazy regimes.

`ohlcv` produces `(open, high, low, close, volume)`. The volume column is the sum
of column-1 values inside each bar.

`ohlcv2` produces `(open, high, low, close, buy_vol, sell_vol)`. Buy volume is
`sum(max(v, 0))` and sell volume is `sum(|min(v, 0)|)` per bar (both non-negative),
the signed-part decomposition of the column-1 signed volume.

## `ohlc_bars` (re-aggregate pre-built OHLC bars)

`ohlc_bars` compresses already-built OHLC bars into wider bars. It requires a
`(T, 4)` input: columns are `[open, high, low, close]`. The bar re-aggregation
applies `first` to column 0, `max` to column 1, `min` to column 2, and `last` to
column 3, preserving the OHLC interpretation across bar boundaries. Both `freq=`
and `count=` bucketing work; the agg is supported in eager, graph, and lazy
regimes without restriction.

## `ohlcv_bars` (re-aggregate pre-built OHLCV bars)

`ohlcv_bars` compresses OHLCV bars with one or more volume columns. It requires a
`(T, N)` input with `N >= 5`: columns `[open, high, low, close, vol1, vol2, ...]`.
The first four columns reduce as in `ohlc_bars`; every trailing column is summed.
Output shape is `(bars, N)`. The number of trailing volume columns may vary freely;
the plan is resolved from the actual input width at runtime. Supported in eager,
graph, and lazy regimes.

## `threshold=` -- cumulative-driver (information bars)

`threshold=T` is the fourth mutually exclusive mode selector. A bar closes when
the cumulative sum of a **driver** stream since the last close reaches `T`. The
observation that crosses the threshold is included in the closing bar; then the
driver accumulator resets to 0.

Typical use: volume bars (each bar contains exactly `T` units of traded volume),
dollar bars (each bar contains `T` notional), or imbalance bars where the driver
is a signed trade-flow measure. Call as
`values, index = Resample(price_vi, driver, threshold=T, agg='ohlc')`.

- `price_vi` is a `(values, index)` tuple (or a bare array with a separate
  index). It supplies the bar's value stream and the time axis.
- `driver` is a 1-D array of the same length as `price_vi`. It is the per-event
  driver that accumulates toward `T`.
- `threshold=T` must be `> 0` (raises `ValueError` otherwise).
- `threshold=` is mutually exclusive with `freq=`, `every=`, and `count=`; passing
  two mode selectors raises `ValueError`.
- A `NaN` driver value is **ignored**: it does not advance the cumulative sum, and
  the clock does not move. The observation's value **does** contribute to the bar's
  reducer (the bar boundary is not affected by the NaN, but the price is still
  accumulated).
- Bar labels: by default (`label="left"`) each bar's index is the actual index of
  the first event in that bar. With `label="right"` it is the index of the last
  event (the crossing observation).
- The trailing partial bar (events since the last close, before cumulative driver
  reaches `T`) is always emitted at end of input.
- All `agg=` forms work with `threshold=`: string aggs (`last`, `ohlc`, etc.) and
  functor aggs (`ExpandingSkew()`, ...).
- Works in eager, `Pipeline` graph, and lazy iterator regimes with identical results.

**Graph regime** -- wire two `Input` nodes, one for the value stream and one for
the driver, then call `Resample(value_node, driver_node, threshold=T, agg=...)`.
For example: `x = Input('price'); d = Input('driver');
node = Resample(x, d, threshold=500, agg='ohlc');
dag = Pipeline([x, d], [node]);
ohlc, bar_idx = dag((price_arr, idx), (volume_arr, idx))`.

<!-- HELP_END -->
## Examples

### Mean bar

Downsample tick values into buckets of width 10 and take the mean of each bucket.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Resample

   idx  = np.array([0, 3, 10, 12, 20])
   vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

   values, index = Resample(freq=10, agg="mean")(vals, idx)
   print(values)
   print(index)
```

### OHLCV bars from tick data

Pass a two-column `[price, volume]` array and use `agg="ohlcv"` for a
labelled five-column bar stream.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Resample
   np.random.seed(0)

   price  = np.array([100., 101., 99., 102., 98., 103., 97., 104., 96., 105.])
   volume = np.array([10., 20., 15., 30., 12., 22., 18., 25., 14., 28.])
   idx    = np.arange(10, dtype=np.int64)

   values, index = Resample(freq=5, agg="ohlcv")(np.column_stack([price, volume]), idx)
   # columns positional: open(0), high(1), low(2), close(3), volume(4)
   print(index)
   print(values.round(2))
```

### Custom per-bar statistic with a functor

Any `EvalOp` functor resets at each bar boundary and accumulates within the
bar. `ExpandingSkew()` returns the intra-bar price skewness at bar close.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import ExpandingSkew, Resample
   np.random.seed(0)

   price = np.random.normal(100, 1, 20)
   idx   = np.arange(20, dtype=np.int64)

   values, index = Resample(freq=5, agg=ExpandingSkew())(price, idx)
   print(values.round(4))
```

### Several statistics over the same bucketing

Run one `Resample` per statistic and align them with `CombineLatest`. Every
`Resample` shares the same `freq=` and clock, so the bars line up exactly.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import ExpandingSkew, ExpandingSlope, Resample, CombineLatest
   np.random.seed(0)

   price = 100 + np.cumsum(np.random.normal(0, 0.3, 20))
   idx   = np.arange(20, dtype=np.int64)

   skew  = Resample(freq=5, agg=ExpandingSkew())(price, idx)
   slope = Resample(freq=5, agg=ExpandingSlope())(price, idx)
   aligned, bar_idx = CombineLatest()(skew, slope)
   print(bar_idx)
   print(aligned.round(4))
```

### Multi-column bars in a `Pipeline`

Inside a graph, place a `Resample` node on each per-tick expression rooted at an
`Input`, then align them with `CombineLatest`. All bars share the same `freq=` and
clock, so they cannot drift. You bind data to the named inputs at call time.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import First, Last, ExpandingMax, ExpandingMin, Resample, CombineLatest
   from screamer import Input, Pipeline

   t_arr  = np.arange(10, dtype=np.int64)
   px     = np.array([100., 101., 99., 102., 98., 103., 97., 104., 96., 105.])

   price = Input("price")
   open_b  = Resample(freq=5, agg=First())(price)
   high_b  = Resample(freq=5, agg=ExpandingMax())(price)
   low_b   = Resample(freq=5, agg=ExpandingMin())(price)
   close_b = Resample(freq=5, agg=Last())(price)
   bars = CombineLatest()(open_b, high_b, low_b, close_b)
   ohlc, ohlc_idx = Pipeline([price], [bars])(price=(px, t_arr))
   print(ohlc_idx)
   print(ohlc.round(2))
```

Use `count=` to bucket by a fixed number of events instead of an index interval.

### Volume bars with `threshold=`

Build OHLC information bars that close once the cumulative traded volume reaches
a threshold. The crossing observation is included in the closing bar.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer.streams import Resample

   price  = np.array([10., 11, 12, 13, 14, 15, 16, 17, 18])
   volume = np.array([ 3.,  4,  2,  5,  1,  6,  2,  3,  4])
   idx    = np.arange(9, dtype=np.int64)

   ohlc, bar_idx = Resample((price, idx), volume, threshold=5, agg='ohlc')
   # columns: open, high, low, close
   for i, k in enumerate(bar_idx):
       print(f"bar {k}: {ohlc[i]}")
```
