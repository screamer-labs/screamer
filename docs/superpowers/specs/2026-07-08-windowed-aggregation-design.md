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

- **string** (existing shorthands: `first/last/min/max/sum/count/mean/ohlc`) maps
  to a builtin C++ reducer. Kept as convenient sugar.
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
- Optionally add thin `Expanding{Sum,Max,Min,Prod}` **aliases** to the `Cum*`
  implementations for a single discoverable prefix. Aliases only, never
  duplicate logic.

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

## Open decisions (confirm before implementation)

- The `Expanding*` set for the first cut. Proposal: `Mean`, `Var`, `Std`,
  `Skew`, `Slope`; add `Kurt` if cheap.
- How a `dict` agg surfaces labelled columns in the return (structured array,
  plain 2D plus a names list, or a Stream carrying column names).
- Whether string aggs stay as sugar or become internal aliases to the builtin
  C++ reducers (recommend: keep the sugar, back it by the builtins).
- Whether to add the `Expanding{Sum,Max,Min,Prod}` aliases now or defer.
- Multi-input reducers as `agg` (2-input masked sum): defer.

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
