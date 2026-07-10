# resample redesign: contextual `freq`, `agg` as a functor, composition for multi-column

**Status:** design approved in discussion; awaiting written-spec review before planning.

**Supersedes** the design portion of the earlier draft `2026-07-10-unified-streaming-stage4-resample-freq.md` (that draft predates the `agg`-as-functor decision and the deletion of the dict form).

## Goal

Make `resample` a small, polymorphic, one-stream windowed reducer that behaves like every other screamer function (`RollingMean`-style: stream in, stream out), and resolve three long-standing complaints:

1. the two badly-named bucket arguments `every=` / `count=`;
2. `resample` overreaching into a graph-like `agg={...}` dict form that referenced `Input` ports (a `Dag` concept) and could not express multi-input reductions such as VWAP;
3. an inconsistency where the positional argument meant *data* in the single-column form but the *clock* in the dict form.

## The model in one sentence

`resample(data, index=None, *, freq, agg="last", origin=0, label="left", fill="skip")` buckets `data` into bars defined by `freq`/`index`, feeds each bar's samples to the `agg` functor, takes `agg`'s output at the bar's last sample, resets `agg` for the next bar, and returns one value (or row) per bar.

`resample` is a plain function: `data` may be a scalar-stream array, a lazy iterator, or a graph `Node`, and the output container mirrors the input (Rule A). It owns the `agg` functor and drives it, so nothing upstream is reset and it works identically eager or streaming.

## Bucketing: one contextual `freq`, read from the index

`every=` and `count=` collapse into a single `freq`, whose meaning is read from the index (as pandas `date_range` reads `freq`):

| index | `freq` type | meaning |
|---|---|---|
| none (`index=None`) | `int` | a bar every `N` events (count mode) |
| integer | `int` | a span of `N` index units, bar `n` = `[origin + n*freq, origin + (n+1)*freq)` |
| `datetime64` / timestamp | offset string (`"1min"`) or `timedelta` | wall-clock bar, converted to integer index units |

Providing an index makes `freq` a span; omitting it makes `freq` a count. There is no separate mode argument and no "index optional maybe there" ambiguity. The C++ core stays pure integer-index space; the offset/`timedelta` conversion is a thin Python-layer feature that validates the `freq` type against the index dtype and raises a clear error on a nonsensical pair (for example a `timedelta` `freq` on an integer index).

**Count mode carrying an index is intentionally dropped.** If you want "every N events", omit the index. (Decision D1.)

## Aggregation: `agg` is a functor (with string synonyms)

`agg` is any single-input functor. `resample` feeds it the bar's samples, reads its value at the bar's last sample, and resets it per bar. Because the reducer is *owned by* `resample` (a parameter, not an upstream stream it samples), `resample` drives and resets it, which is why per-bar reduction works on plain eager arrays as well as on lazy iterators and graph nodes.

The canonical reducers are the `Expanding*` family, because they are cumulative from reset: reset per bar turns them into exactly the bar statistic, and the value at bar-end is the full-bar result.

**Short string synonyms** are provided for the popular choices, and the full functor syntax is always accepted:

| string | functor | bar meaning |
|---|---|---|
| `"sum"` | `ExpandingSum()` | sum |
| `"mean"` | `ExpandingMean()` | average |
| `"max"` | `ExpandingMax()` | high |
| `"min"` | `ExpandingMin()` | low |
| `"first"` | `First()` | open |
| `"last"` (default) | `Last()` | close / sample at bar-end |
| `"std"` | `ExpandingStd()` | per-bar volatility |
| `"var"` | `ExpandingVar()` | variance |
| `"count"` | (event count in the bar) | number of ticks |
| `"ohlc"` | multi-output | open, high, low, close (four columns, single pass) |

Further synonyms backed by existing functors, offered as easy additions: `"prod"` (`ExpandingProd`), `"sem"`, `"skew"` (`ExpandingSkew`), `"kurt"` (`ExpandingKurt`), `"argmax"`/`"argmin"` (the index/time of the bar's high/low). The string set is a small closed vocabulary (researched against pandas resample, the SQL standard, and QuestDB); anything outside it is passed as a functor.

```python
resample(price, index=t, freq="1min", agg="ohlc")     # string synonym
resample(price, index=t, freq="1min", agg=ExpandingStd())   # full functor
resample(vol,   index=t, freq="1min", agg="sum")
```

### Semantics and caveats

- **Value at bar-end.** The bar value is `agg`'s output at the bar's last sample. Only cumulative reducers (`Expanding*`, `First`, `Last`, `count`) give textbook bar statistics. A non-cumulative functor (`RollingMean(2)`, `Diff`, an EMA) yields a defined-but-niche value; a functor whose warmup is longer than the bar (`RollingMean(5)` in a three-tick bar) yields NaN, since it resets each bar. This is allowed and predictable, not the headline use.
- **Single-input only.** Multi-input functors (`BOP`, a two-argument VWAP) are not valid `agg` values; those are composition (below).
- **Reset is only-outer.** If a composite functor is passed as `agg`, `resample` resets the outer functor's own state; a stateless inner (`Mul`, `PosPart`) has a no-op reset, and a stateful inner persists across bars. Recommended practice: keep `agg` a single functor and do transforms upstream.
- **Multi-output functor** (`ohlc`) produces several columns in one pass without any dict.
- **Empty bars** are governed by `fill` (`skip` / `nan` / `carry`), unchanged.
- All functors have `reset()` (it is a pure-virtual on `EvalOp`; stateless functors get a no-op default), so any functor is a legal `agg`.

## Multi-column and multi-input are composition, not `resample` features

Multiple statistics of different streams are separate one-stream resamples combined by `combine_latest` (they align on the shared bar grid because the bar labels match):

```python
o = resample(price, index=t, freq="1min", agg="first")
h = resample(price, index=t, freq="1min", agg="max")
l = resample(price, index=t, freq="1min", agg="min")
c = resample(price, index=t, freq="1min", agg="last")
v = resample(vol,   index=t, freq="1min", agg="sum")
bars = combine_latest(o, h, l, c, v)          # or: combine_latest(resample(price, ..., agg="ohlc"), v)
```

A statistic *over* several streams (VWAP, spread, correlation) combines the inputs before and/or the bar outputs after; `resample` itself stays one-input:

```python
vwap = resample(price * vol, index=t, freq="1min", agg="sum") \
     / resample(vol,         index=t, freq="1min", agg="sum")
```

This is why the `agg={...}` dict/ports form is deleted: everything it did is either a multi-output single functor (`ohlc`) or ordinary composition, and it was the only part of `resample` that behaved like a `Dag`.

## What this deletes (breaking, pre-1.0)

- `every=` and `count=` on `resample`, replaced by contextual `freq=`.
- The `agg={dict}` multi-column form and its `Input`-port reducer expressions (`_split_reducer_expr`, the graph `multi_resample` front-end via `resample`). The low-level multi-column node may remain as an internal optimization target but is no longer a public `resample` shape.
- Count-mode carrying an index (no `freq=` spelling; omit the index instead).

## Design constraints this satisfies

- **input type maps to output type / polymorphic:** `resample` dispatches on `data` like any functor: array to array, list to list, lazy iterator to lazy iterator, scalar N/A (a single event cannot form a bar), `Node` to `Node`.
- **a `Dag` is indistinguishable from a function:** `resample` is a plain function usable inside a `Dag` (with `Input`s) or on concrete data, identically. It does not itself reference `Input`.
- **index types (None / int / datetime64):** handled by the contextual `freq` rule; the index is ordinary input data.
- **reset:** internal to the owned `agg` functor; no upstream or "live input" reset, so it holds for all input kinds.
- **causality / streaming:** every canonical `agg` is an O(1) incremental reducer; `batch == lazy` is preserved because the engine and its numeric output are unchanged (only the argument surface changes).

## Migration surface

Roughly 158 `every=` and 36 `count=` call sites across source, 14 test files, 4 notebooks, and 2 doc pages, plus any `agg={dict}` usage. Hard cutover (Decision D3): remove `every=`/`count=`/`agg={dict}`, add `freq=` and `agg=<functor|string>`, and migrate every call site in the same change set.

## Out of scope (separate later stage)

- `(index, NaN)` heartbeats and retiring `advance()` / `dag.live()` (the live/clock path) - their own stage.
- `median` / `quantile` as buffering aggs (not O(1); a bar is bounded so they are feasible, but they break the streaming property - offer later, documented, or approximate).
- `mode` / `nunique` (need a set or histogram; niche for continuous prices).

## Testing strategy

- **Equivalence oracle:** each `freq=` call produces byte-identical values and index to the `every=`/`count=` call it replaces (the engine is unchanged). Each `agg="name"` equals `agg=<the mapped functor>`.
- **Per-bar reset:** `resample(x, count=N, agg=ExpandingSum())` yields per-bar sums, not cumulative (already verified: three 3-count bars of `1..9` give `[6, 15, 24]`, not `[6, 21, 45]`).
- **Composition:** multi-column `combine_latest` bars and the VWAP composition match a hand-computed oracle.
- **batch == lazy:** the whole surface keeps the Stage 2/3 guarantee (`resample(array)` vs `list(resample(generator))`).
- **Dispatch:** array to array, generator to lazy iterator, `Node` to node; string synonyms resolve to the mapped functors; a multi-input functor passed as `agg` raises a clear error.
