# Windowed Aggregation, Expanding Statistics, and Signed-Part Helpers

**Status:** design (pre-implementation)
**Date:** 2026-07-08

## Goal

Turn `resample` from a fixed-aggregation downsampler into a general
windowed-aggregation operator, so arbitrary per-bar statistics (OHLCV, buy/sell
volume, trend, skew, and anything else) are expressible without special-casing.
Add the building blocks that need: an `Expanding*` statistic family and
`PosPart`/`NegPart` element-wise helpers.

## Governing principle: logic in C++, thin Python

All numeric logic lives in the C++ core. Python (and any future binding) is a
thin marshalling and dispatch layer with no compute loops. This is a hard
constraint, for two reasons:

1. Pure-C++ consumers must get the full functionality without Python.
2. A planned JavaScript/WASM binding must reach the same functionality by
   re-implementing only the thin shim, never the numerics.

Concretely, this design must **retire the Python `_ResampleAccum` and the eager
bucketing loop in `screamer/streams.py`** and route both the eager (array) and
graph (DAG) paths through the same C++ windowing engine. Today the graph path is
a C++ push-node while the eager path loops in Python; that split violates the
principle and will not port to WASM.

Every new functor is registered under `EvalOp` so the C++ engine can hold and
drive it, and so bindings expose it uniformly.

## Design

### 1. `resample(agg = str | functor | dict)`

`agg` accepts three forms, all resolved to a C++ reducer:

- **string** shorthands map to a builtin C++ reducer. Existing:
  `first/last/min/max/sum/count/mean/ohlc`. New:
  - `ohlcv`  -> `[open, high, low, close, volume]` (volume = sum of the volume input).
  - `ohlcv2` -> `[open, high, low, close, buy_vol, sell_vol]`, where
    `buy_vol = sum of positive volume` and `sell_vol = sum of |negative volume|`.

  `ohlc` reads one price stream. `ohlcv` and `ohlcv2` are **multi-input**: the
  `values` argument is a 2-column stream (`[price, volume]` for `ohlcv`,
  `[price, signed_volume]` for `ohlcv2`), using the existing "(T, N) array as N
  inputs" model. Their output columns are labelled (see "Labelled output").
- **functor** (any `EvalOp`): the C++ windowing node feeds every in-bar sample to
  the functor, samples its output at bar close, then calls `reset()`. Supports
  N->M reducers (for example `ohlc` is effectively 1->4; a custom reducer may be
  1->M).
- **dict** `{name: agg}`: several reducers over the same bucketing, emitted as one
  labelled multi-column bar record. Column names come from the keys.

Semantics are unchanged from today, just generalized: causal, emit at bar close,
reset the reducer each bar, only non-empty buckets emit, trailing partial bucket
flushed at end of input, NaN ignored. Bucketing by `every=` (index interval) or
`count=` (event count) is unchanged.

Requirement on any functor used as a reducer: expose `reset()` (functors already
do) and be callable one sample at a time (they are).

**Key subtlety (drives the Expanding family):** whole-bar statistics need
*expanding* reducers, not fixed-window rolling ones. `RollingSkew(20)` reset per
bar only sees the last 20 in-bar samples, not the whole bar. So the natural
reducers for per-bar stats are the `Expanding*` family below.

#### Labelled output for multi-column aggs

`ohlc`, `ohlcv`, `ohlcv2`, and `dict` aggs produce a multi-column bar matrix with
named columns. How the names travel:

- The C++ node returns a plain 2-D `float64` bar matrix plus the column labels as
  strings (and the bar index). Plain `float64` keeps the frame composable: you
  can do matrix math on it or feed a column into another functor. No structured
  arrays (awkward for math) and no pandas (a runtime dependency, and not
  WASM-portable).
- Labels travel **with the data** on the `Stream` type: extend `Stream` with an
  optional `columns` tuple. A multi-column resample returns a `Stream` whose
  `.values` is 2-D, `.index` is the bar labels, and `.columns` are the names.
  Named access via `stream["buy_vol"]` (and `.column("buy_vol")`), returning a
  1-D view. Column order is the dict insertion order, or the fixed OHLC(V) order.
- The JS/WASM binding exposes the same `(matrix, index, names)`; the labelled
  container is a thin shim with no compute. (A distinct `Frame` type is an
  option instead of extending `Stream`, but `Stream` with 2-D values already *is*
  a frame indexed by bar label, so reuse it and avoid a new public type.)

Sub-decision (pre-1.0, low stakes): whether *all* `resample` returns unify on
`Stream`, or only the multi-column aggs return a labelled `Stream` while scalar
aggs keep returning a plain `(values, index)`. Recommend unifying on `Stream` for
one return type across the raw / Stream / Node regimes.

### 2. `Expanding*` statistic family

New C++ functors that accumulate from `reset()` over all samples seen, with no
window cap: `ExpandingMean`, `ExpandingVar`, `ExpandingStd`, `ExpandingSkew`,
`ExpandingKurt`, `ExpandingSlope` (linear-regression slope / trend).

They reuse the same moment-accumulation math as the `Rolling*` / `Ew*` versions
without the sliding-window eviction. They are independently useful (an expanding
window over a whole series, matching pandas `.expanding()`), not only inside
bars. Naming matches pandas so a quant finds them by the word they already use.

### 3. `Cum*` vs `Expanding*` naming decision

`CumSum` / `CumMax` / `CumMin` / `CumProd` already exist (numpy vocabulary) and
are exactly the expanding sum / extrema / product. Decision:

- **Keep** `Cum*` for those four (`cumsum` is too well-known to drop).
- **Add** `Expanding*` for the moment stats and slope (`ExpandingMean` reads far
  better than `CumMean`).
- Add thin `Expanding{Sum,Max,Min,Prod}` **aliases** to the `Cum*`
  implementations, giving one discoverable `Expanding*` prefix alongside the
  well-known `Cum*` names. Aliases only, never duplicate logic.

### 4. `PosPart` / `NegPart`

New element-wise C++ functors: `PosPart(x) = max(x, 0)` and
`NegPart(x) = max(-x, 0)`. `PosPart` is `Relu` under a quant-readable name;
share the C++ implementation (make one an alias of the other) rather than
duplicating. `NegPart` supplies the negative-part magnitude.

These make signed splits a one-line recipe with no dedicated accumulators. For
signed volume (buy positive, sell negative):

```python
buy_vol  = resample(PosPart()(signed_vol), every=BAR, agg="sum")
sell_vol = resample(NegPart()(signed_vol), every=BAR, agg="sum")
```

The same trick covers up/down tick counts, positive/negative returns, order-flow
imbalance, and so on. The more general case (side as a separate signal) is a
masked sum, `resample(size * (side > 0), agg="sum")`, and can grow a named
`SumWhere` later if it proves common; it is out of scope here.

## What moves to / lives in C++

- The windowing + reduce + reset + emit engine (the `resample` node): C++, shared
  by eager and graph paths. Replaces the Python `_ResampleAccum` and eager loop.
- The builtin string reducers: C++.
- `Expanding*`, `PosPart`, `NegPart`: C++ functors registered under `EvalOp`.
- Python does only: resolve `agg` (string -> builtin id; functor -> pass the C++
  op handle; dict -> build a multi-reducer node plus column labels), dispatch
  raw / Stream / Node, marshal arrays. No numeric loops.

## Decisions

Settled:

- `Expanding*` first cut: `Mean`, `Var`, `Std`, `Skew`, `Kurt`, `Slope`.
- `Expanding{Sum,Max,Min,Prod}` aliases to `Cum*`: yes.
- String aggs: keep as sugar, backed by the builtin C++ reducers.
- New string aggs: `ohlcv`, `ohlcv2` (multi-input, see above).
- Labelled multi-column output: plain 2-D `float64` matrix + labels carried on
  `Stream.columns`; no pandas, no structured arrays (see "Labelled output").

Still open:

- Unify *all* `resample` returns on `Stream`, or only the multi-column aggs
  (recommend: unify on `Stream`).
- General masked sum as a 2-input reducer (`SumWhere`): defer past this cut.

## Implementation phases (behavior-preserving first)

1. **C++ engine unification.** Move the resample bucketing + reduce + reset +
   emit into C++; route eager and graph through it. String reducers become C++
   builtins. No API change; `batch == stream == oracle` tests stay green.
2. **Functor reducers.** `agg` accepts any `EvalOp`; support N->M reducers.
3. **New functors.** `Expanding*`, `PosPart`/`NegPart`, optional `Cum*` aliases;
   register under `EvalOp`.
4. **Python thin layer.** `resample(agg = str | functor | dict)` dispatch plus
   dict -> multi-reducer with labels; export new functors via the init generator.
5. **Docs and tests.** Frontmatter pages + `topics` + `nan_policy` for each new
   functor; updated `resample` page; a buy/sell-vol and custom-bar recipe
   notebook; `Expanding*` verified against pandas `.expanding()`; functor/dict
   aggs verified `batch == stream`.

Each new functor follows the existing add-a-function checklist (C++ + binding +
docs frontmatter + topics + nan_policy + baseline).

## WASM / pure-C++ note

Because all logic lands in C++ under `EvalOp` and is driven by the C++ engine,
the pure-C++ library and a future JS/WASM binding get identical functionality;
the Python layer is a thin shim reproducible in JavaScript with no numerics.
