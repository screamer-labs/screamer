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
per-bar aggregation, and return a labelled `Stream`. A bucket emits only once a
later index proves it complete; the trailing partial bucket emits at end of input.
Usable eagerly (raw arrays or `Stream`) and inside a `Dag`.

Feeding a lazy iterator of `(value, index)` pairs returns a lazy iterator of bar events; feeding arrays or a `Stream` returns the batch result.

The `agg` parameter accepts two forms:

**String shorthand**: one of `first`, `last`, `min`, `max`, `sum`, `count`,
`mean`, `ohlc`, `ohlcv`, `ohlcv2`. `ohlc` returns four columns
(`open`, `high`, `low`, `close`). `ohlcv` and `ohlcv2` accept a two-column
input `[price, volume]`; see below.

**Any `EvalOp` functor**, e.g. `ExpandingSkew()`. The functor is `reset()` at
each bar boundary and fed every in-bar sample; its last output before the close
is emitted as the bar value. All screamer functors are valid reducers.

**Several reducers at once**: run one `Resample` per statistic over the same
bucketing and align the results with `CombineLatest`. Because every `Resample`
shares the same `freq=` (or `count=`) and clock, the bars line up exactly and
cannot drift, e.g.
`CombineLatest()(Resample(freq=5, agg="first")(price, t), Resample(freq=5, agg="sum")(vol, t))`.
Inside a `Dag`, place each per-stat `Resample` node on its upstream expression
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
interval with no ticks). It applies to eager arrays, `Stream`s, and graphs alike;
it is not `Dag`-only.

- `"skip"` (default): emit no row for an empty bucket (the legacy behavior).
- `"nan"`: emit an all-NaN row at the empty bucket's label.
- `"carry"`: repeat the previous emitted row's values verbatim.

Only **internal** empty buckets (gaps between two events) are filled by `Resample`
itself. Trailing empty buckets after the last event are not synthesized here; that
needs a clock, via either [`dag.live().advance(now)`](#dag-live) or a clock input
wired into a `Dag`. The [`dag.live()` reference](#dag-live) has a worked example.

`fill=` is meaningful only under `freq=`. With `count=`, a bar is defined by
holding `N` events, so empty bars cannot exist by construction and `fill=` has no
effect.

### `fill=` edge cases

- **Leading buckets** (before the first event) are never synthesized on the eager
  path: output starts at the bucket containing the first tick. A tick at index 47
  with `freq=10` starts at label 40, not 0. Leading empty buckets arise only from a
  clock (a `Dag` clock input or [`advance()`](#dag-live)) that advances time before
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

## Labelled output and `Stream.columns`

Every `Resample` call returns a `Stream`, which is also unpackable as
`(values, index)` for backward-compatible tuple unpacking. Multi-column
aggregations (`ohlc`, `ohlcv`, `ohlcv2`) set `.columns` on the returned
`Stream` to a tuple of column names; single-value aggregations leave `.columns`
as `None`. Use `bars["close"]` to read a named 1-D column, or iterate
`(values, index) = bars` for the full array.

## `ohlcv` and `ohlcv2` (two-column input)

Both require `values` to be a `(T, 2)` array: column 0 is price, column 1 is
volume (unsigned for `ohlcv`, signed for `ohlcv2`).

`ohlcv` produces `(open, high, low, close, volume)`. The volume column is the sum
of column-1 values inside each bar.

`ohlcv2` produces `(open, high, low, close, buy_vol, sell_vol)`. Buy volume is
`sum(PosPart(signed_vol))` and sell volume is `sum(NegPart(signed_vol))` per bar,
the signed-part decomposition.

<!-- HELP_END -->

## Examples

### Mean bar

Downsample tick values into buckets of width 10 and take the mean of each bucket.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Resample
   # --- hide: stop ---
   idx  = np.array([0, 3, 10, 12, 20])
   vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

   bars = Resample(freq=10, agg="mean")(vals, idx)
   print(bars.values)
   print(bars.index)
```

### OHLCV bars from tick data

Pass a two-column `[price, volume]` array and use `agg="ohlcv"` for a
labelled five-column bar stream.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Resample
   np.random.seed(0)
   # --- hide: stop ---
   price  = np.array([100., 101., 99., 102., 98., 103., 97., 104., 96., 105.])
   volume = np.array([10., 20., 15., 30., 12., 22., 18., 25., 14., 28.])
   idx    = np.arange(10, dtype=np.int64)

   bars = Resample(freq=5, agg="ohlcv")(np.column_stack([price, volume]), idx)
   print(bars.columns)
   print(bars.values.round(2))
```

### Custom per-bar statistic with a functor

Any `EvalOp` functor resets at each bar boundary and accumulates within the
bar. `ExpandingSkew()` returns the intra-bar price skewness at bar close.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import ExpandingSkew, Resample
   np.random.seed(0)
   # --- hide: stop ---
   price = np.random.normal(100, 1, 20)
   idx   = np.arange(20, dtype=np.int64)

   bars = Resample(freq=5, agg=ExpandingSkew())(price, idx)
   print(bars.values.round(4))
```

### Several statistics over the same bucketing

Run one `Resample` per statistic and align them with `CombineLatest`. Every
`Resample` shares the same `freq=` and clock, so the bars line up exactly.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import ExpandingSkew, ExpandingSlope, Resample, CombineLatest
   np.random.seed(0)
   # --- hide: stop ---
   price = 100 + np.cumsum(np.random.normal(0, 0.3, 20))
   idx   = np.arange(20, dtype=np.int64)

   skew  = Resample(freq=5, agg=ExpandingSkew())(price, idx)
   slope = Resample(freq=5, agg=ExpandingSlope())(price, idx)
   aligned, bar_idx = CombineLatest()(skew, slope)
   print(bar_idx)
   print(aligned.round(4))
```

### Multi-column bars in a `Dag`

Inside a graph, place a `Resample` node on each per-tick expression rooted at an
`Input`, then align them with `CombineLatest`. All bars share the same `freq=` and
clock, so they cannot drift. You bind data to the named inputs at call time.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import First, Last, ExpandingMax, ExpandingMin, Resample, CombineLatest
   from screamer.dag import Input, Dag
   # --- hide: stop ---
   t_arr  = np.arange(10, dtype=np.int64)
   px     = np.array([100., 101., 99., 102., 98., 103., 97., 104., 96., 105.])

   price = Input("price")
   open_b  = Resample(freq=5, agg=First())(price)
   high_b  = Resample(freq=5, agg=ExpandingMax())(price)
   low_b   = Resample(freq=5, agg=ExpandingMin())(price)
   close_b = Resample(freq=5, agg=Last())(price)
   bars = CombineLatest()(open_b, high_b, low_b, close_b)
   ohlc, ohlc_idx = Dag([price], [bars])(price=(px, t_arr))
   print(ohlc_idx)
   print(ohlc.round(2))
```

Use `count=` to bucket by a fixed number of events instead of an index interval.
