---
name: resample
title: resample
kind: function
short: Causal windowed downsample of a 1-D or multi-column value stream.
topics:
- streams
covers:
- resample_iter
---

# `resample`

Causal windowed downsampling. Group a stream into fixed index-interval buckets
(`every=`) or fixed event-count buckets (`count=`), reduce each bucket with a
per-bar aggregation, and return a labelled `Stream`. A bucket emits only once a
later index proves it complete; the trailing partial bucket emits at end of input.
Usable eagerly (raw arrays or `Stream`) and inside a `Dag`.

The `agg` parameter accepts three forms:

**String shorthand** -- one of `first`, `last`, `min`, `max`, `sum`, `count`,
`mean`, `ohlc`, `ohlcv`, `ohlcv2`. `ohlc` returns four columns
(`open`, `high`, `low`, `close`). `ohlcv` and `ohlcv2` accept a two-column
input `[price, volume]`; see below.

**Any `EvalOp` functor** -- e.g. `ExpandingSkew()`. The functor is `reset()` at
each bar boundary and fed every in-bar sample; its last output before the close
is emitted as the bar value. All screamer functors are valid reducers.

**Dict** `{name: agg}` -- run several reducers over the same bucketing in one
call, returning a labelled `Stream` whose `.columns` are the dict keys (insertion
order). Each entry produces one column. There are two forms:

- *Eager*, over raw arrays or a `Stream`: each value is a string or functor
  reducer applied to the single value stream, e.g.
  `{"open": "first", "vol": "sum"}`.
- *Graph / lazy*, inside a `Dag` (`resample(t, agg={...})` where the dict values
  are `Node` expressions): each value is a lazy `Reducer()(sub_expr)` -- its top
  node is the per-bar reducer and its single input is the upstream port, so
  per-tick transforms live in the expression, e.g.
  `{"buy": ExpandingSum()(PosPart()(vol))}`. All columns share one bar clock and
  cannot drift; the first positional argument `t` is the clock, and data binds at
  call time.

## Bucketing: `every=` vs `count=`

Exactly one of `every=` or `count=` sets how bars are bounded, and they answer
different questions.

- `every=W` buckets along the **index**. Bar `n` is the half-open interval
  `[origin + n*W, origin + (n+1)*W)`, so the index values decide membership. Bars
  have equal width on the index but a variable number of ticks; a tick exactly on a
  boundary starts the later bar. Boundaries are anchored at `origin` (default `0`,
  i.e. multiples of `W`), **not** at the first tick -- set `origin=` to shift the
  grid. Internal empty intervals are real and controlled by `fill=`.
- `count=N` buckets by **arrival order**. A bar closes every `N` events and never
  consults the index values to place boundaries. Bars have an equal number of ticks
  but a variable width on the index, and one bar can straddle an arbitrary index gap
  without noticing it.

The `index` argument is **optional in both modes**. `count=` does not need it to
find boundaries; `every=` uses it as the timeline being bucketed. If omitted, row
position (`0, 1, 2, ...`) is used as the index.

Bar **labels** depend on the mode and on `label=`:

- `every=`: the bar's **grid edge** -- `origin + n*W` for `label="left"` (default)
  or `origin + (n+1)*W` for `label="right"`. This is the interval boundary itself,
  which need not equal any actual tick's index (and a right label can sit past the
  last tick).
- `count=`: an **actual tick index** -- the **first** tick of the bar for
  `label="left"`, the **last** for `label="right"`.

Concretely, eight ticks at index `[0, 1, 2, 10, 11, 20, 21, 22]`:

- `every=10` -> bars `{0,1,2} {10,11} {20,21,22}` (counts 3, 2, 3), labels
  `[0, 10, 20]` (grid edges).
- `count=3`  -> bars `{0,1,2} {10,11,20} {21,22}` (counts 3, 3, 2), labels
  `[0, 10, 21]` (first tick of each bar). The middle bar straddles the `11 -> 20`
  gap because `count=` measures rows, not index distance.

## Empty buckets: `fill=`

`fill=` controls what happens when an `every=` bar contains no samples (an index
interval with no ticks). It applies to eager arrays, `Stream`s, and graphs alike --
it is not `Dag`-only.

- `"skip"` (default) -- emit no row for an empty bucket (the legacy behavior).
- `"nan"` -- emit an all-NaN row at the empty bucket's label.
- `"carry"` -- repeat the previous emitted row's values verbatim.

Only **internal** empty buckets (gaps between two events) are filled by `resample`
itself. Trailing empty buckets after the last event are not synthesized here; that
needs a clock, via `advance()` or a clock input on a `Dag` (see the custom-bars
notebook).

`fill=` is meaningful only under `every=`. With `count=`, a bar is defined by
holding `N` events, so empty bars cannot exist by construction and `fill=` has no
effect.

## Labelled output and `Stream.columns`

Every `resample` call returns a `Stream`, which is also unpackable as
`(values, index)` for backward-compatible tuple unpacking. Multi-column
aggregations (`ohlc`, `ohlcv`, `ohlcv2`, dict) set `.columns` on the returned
`Stream` to a tuple of column names; single-value aggregations leave `.columns`
as `None`. Use `bars["close"]` to read a named 1-D column, or iterate
`(values, index) = bars` for the full array.

## `ohlcv` and `ohlcv2` (two-column input)

Both require `values` to be a `(T, 2)` array: column 0 is price, column 1 is
volume (unsigned for `ohlcv`, signed for `ohlcv2`).

`ohlcv` produces `(open, high, low, close, volume)`. The volume column is the sum
of column-1 values inside each bar.

`ohlcv2` produces `(open, high, low, close, buy_vol, sell_vol)`. Buy volume is
`sum(PosPart(signed_vol))` and sell volume is `sum(NegPart(signed_vol))` per bar
-- the signed-part decomposition.

<!-- HELP_END -->

```{eval-rst}
.. autofunction:: screamer.streams.resample
.. autofunction:: screamer.streams.resample_iter
```

## Examples

### Mean bar

Downsample tick values into buckets of width 10 and take the mean of each bucket.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import resample
   # --- hide: stop ---
   idx  = np.array([0, 3, 10, 12, 20])
   vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

   bars = resample(vals, index=idx, every=10, agg="mean")
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
   from screamer.streams import resample
   np.random.seed(0)
   # --- hide: stop ---
   price  = np.array([100., 101., 99., 102., 98., 103., 97., 104., 96., 105.])
   volume = np.array([10., 20., 15., 30., 12., 22., 18., 25., 14., 28.])
   idx    = np.arange(10, dtype=np.int64)

   bars = resample(np.column_stack([price, volume]), idx, every=5, agg="ohlcv")
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
   from screamer import ExpandingSkew
   from screamer.streams import resample
   np.random.seed(0)
   # --- hide: stop ---
   price = np.random.normal(100, 1, 20)
   idx   = np.arange(20, dtype=np.int64)

   bars = resample(price, idx, every=5, agg=ExpandingSkew())
   print(bars.values.round(4))
```

### Labelled multi-column bars with dict `agg`

Pass a dict to run several reducers over the same bucketing in one call. Each
entry must produce one output column; the keys become `.columns` of the result.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import ExpandingSkew, ExpandingSlope
   from screamer.streams import resample
   np.random.seed(0)
   # --- hide: stop ---
   price = 100 + np.cumsum(np.random.normal(0, 0.3, 20))
   idx   = np.arange(20, dtype=np.int64)

   bars = resample(price, idx, every=5,
                   agg={"skew": ExpandingSkew(), "slope": ExpandingSlope()})
   print(bars.columns)
   print(bars.values.round(4))
```

### Multi-column bars in a `Dag` with a lazy dict

Inside a graph, the dict values are lazy expressions rooted at `Input`s. One
`resample` call builds a single bar node (one clock, N labelled columns) that you
place in a `Dag` and bind to data at call time. All columns share the clock, so
they cannot drift.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import First, Last, ExpandingMax, ExpandingMin
   from screamer.streams import resample
   from screamer.dag import Input, Dag
   # --- hide: stop ---
   t_arr  = np.arange(10, dtype=np.int64)
   px     = np.array([100., 101., 99., 102., 98., 103., 97., 104., 96., 105.])

   price, t = Input("price"), Input("t")
   bars = resample(t, every=5, agg={
       "open":  First()(price),
       "high":  ExpandingMax()(price),
       "low":   ExpandingMin()(price),
       "close": Last()(price),
   })
   ohlc = Dag([t, price], [bars])(t=(t_arr.astype(float), t_arr), price=(px, t_arr))
   print(ohlc.columns)
   print(ohlc.values.round(2))
```

Use `count=` to bucket by a fixed number of events instead of an index interval.
