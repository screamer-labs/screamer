# Multi-stream foundation — the timed-event data model

**Status:** design approved, ready for planning
**Date:** 2026-07-02
**Scope:** Layer 1 of 3 (foundation only). Combinators beyond the minimal set,
and the computational DAG framework, get their own specs on top of this one.

## Motivation

screamer's current model is a stream of bare `double`s aligned **positionally**:
"row `i` of input A pairs with row `i` of input B", and "axis 0 is time" where
*time* is only the row index, never a value. Every multi-input functor
(`RollingCorr`, `RollingSpread`, …) relies on this lockstep assumption.

The moment two streams do not tick together — different rates, asynchronous
arrival, missing samples — positional pairing is wrong. Batch mode needs an
explicit time column to align on; streaming mode needs event-time alignment and
a firing discipline (when do you emit when inputs arrive at different times?).
Neither exists today, and the planned DAG framework cannot be built until there
is a data model that carries an ordering and defines alignment.

This spec defines that foundation: how an ordering key is attached to values,
how streams are combined by that key, and how stored series are turned into
event streams — transparently across batch, streaming, and replay, with results
that are identical in all three modes.

## Non-goals (deferred to later specs)

- **The computational DAG framework** (define nodes + wiring, materialize,
  execute with multiple input/intermediate/output streams). Built on this layer.
- **The full combinator catalogue.** This spec fixes the *minimal* set and their
  contracts; more combinators are additive later.
- **C++ fast-path vs Python-fallback for arbitrary key dtypes.** An
  implementation concern, not part of the contract. The contract only requires
  ordering (and, for wall-clock replay, subtraction) on the key.

## Core principles

These are load-bearing and must also be stated in the user-facing docs
(see "Documentation" below).

### P1. The order key

Every stream has an **order key**, supplied by the user in whatever form they
already have (numpy `datetime64`/`int64`/`float64`, pandas `DatetimeIndex`,
native Python orderables). The library treats it as **opaque**: it only *orders*
and *tests equality*; it never interprets or converts it.

The key is either:
- **explicit** — a timestamp/ordering value the user provides, or
- **implicit** — arrival order for live events; **row number** for arrays with
  no time column.

Two capability tiers:
- **Comparable key** (can be *ordered*) — the weakest requirement; covers row
  numbers and arbitrary orderables. Enables merging, alignment, and backtest
  (max-speed) replay.
- **Metric key** (differences are *meaningful* — `datetime64`, `int64` ns, float
  seconds) — additionally enables **wall-clock replay**, because a sleep
  duration only exists when key-deltas convert to time.

**Consequence:** the current lockstep behaviour is exactly the degenerate
"no key → row index" case. Nothing that exists today changes; it becomes the
floor of the new model.

### P2. Compute is shape-preserving; combinators change cardinality

| Layer | Cardinality | Time / NaN handling |
|---|---|---|
| **Compute functors** (`RollingMean`, `RollingCorr`, `FillNa`, `Ffill`, …) | **Preserved** — output length == input length | NaN handled *internally* by the existing `"ignore"` policy; **no change to these classes** |
| **Combinators** (`merge`, `combine_latest`, `resample`, `dropna`, `filter`, `split`, `pace`) | **May change** — merge/drop/resample all change it | Own all time alignment and stream shaping |

`fillna`/`ffill` are shape-preserving, so they belong to both worlds.
`dropna`/`filter` are cardinality-changing, so they are combinator-only and new.

### P3. Alignment is separate from computation

Time-aware combinators do all key handling and produce *aligned* streams.
Compute functors stay lockstep and consume aligned input. The idiom is:

```python
RollingCorr(window_size=20)(combine_latest(price_a, price_b))
```

Functors never learn about time. Alignment is reusable, inspectable, and the
DAG (later) becomes exactly *alignment edges + compute nodes*.

### P4. Causality and cross-mode identity (hard invariants)

- **Every function is causal** — output at step `t` depends only on inputs at
  steps `≤ t`. **No backward-fill (`bfill`) or any lookahead operator, ever.**
- **Batch == streaming == replay.** The same data produces byte-identical
  numeric results in every mode. Cardinality-changing ops must decide per-event
  from current values only, so batch and streaming drop/keep the identical
  events. Guarded by tests, as `tests/test_stream_vs_batch.py` and
  `tests/test_stream_vs_generator.py` already do for the value path.
- **Pacing never touches values** — only *when* they are emitted. Backtest,
  wall-clock replay, and true-live therefore agree numerically.

## The data model

A **timed stream** is a sequence of `(key, value)` events, surfaced in three
carriers that all route through the same alignment logic:

| Mode | Carrier | Key source |
|---|---|---|
| **Streaming (async, no time)** | live push source / iterator of bare values | arrival order (implicit) |
| **Streaming (explicit time)** | iterator of `(key, value)` events, individually key-sorted | explicit key |
| **Batch (numpy/pandas)** | pandas `Series`/`DataFrame` (index = key); numpy `(keys, values)`; or bare array | index/column if present, else **row number** |

The library detects the carrier and adapts — the "it just works" behaviour —
extending the existing polymorphic dispatch to the multi-stream case.

## The minimal combinator set

### `merge(*sources)`
K-way, order-preserving interleave of individually-sorted sources into **one**
key-sorted stream of **tagged** events (source identity + value). This is the
"pool of next items, emit the earliest" iterator. Use it to feed a single
processor that dispatches by event type.

### `combine_latest(*sources, func=None, emit="when_all", dropna=None, fillna=None)`
The as-of latest-value join. Emits when *any* input advances, carrying each
input's most recent value (this carry is forward-fill — the same mechanism as
`Ffill`). Output key is the triggering event's key. Optional `func` maps the
aligned tuple to a derived value.

- **`emit="when_all"` (default)** — suppress output until every input has
  produced at least one value; first emitted key is the first key at which all
  inputs are warm. Clean by default; never punches holes mid-stream.
- **`emit="on_any"`** — fire on the very first event; not-yet-seen inputs are
  `NaN`. Preserves one-output-per-triggering-event and matches the existing
  warmup-`NaN` convention.

### `filter(source, predicate)` and `dropna(source, how="any", subset=None)`
`filter` is the one real cardinality-reducing primitive. `dropna` is the named
convenience `filter(not-nan)`: on an aligned multi-value stream, `how="any"`
(default) drops the event if any component is `NaN`; `how="all"` only if all
are. `how="any"` is the natural partner to `emit="when_all"` — same
clean-by-default story. Causal and batch==stream exact.

### `split(source, key_func)`
Route one stream into several sub-streams by a key function (the inverse of
`merge`).

### `resample(source, clock)`
Align a stream to a reference clock/grid. Cardinality-changing; causal
(emits a grid point from data at keys `≤` that grid point).

### `pace(source, speed=1.0)` — the driver
Wraps an ordered stream and sleeps between events proportional to key-deltas,
scaled by `speed` (`1` = real time, `10` = 10×, `inf` = no sleep). Requires a
**metric key** (P1). Maps onto the async path (`LazyAsyncIterator` +
`asyncio.sleep`).
- **Backtest driver** = `merge(*series)` with no pacing (sync, max speed).
- **Wall-clock replay** = `pace(merge(*series), speed=…)` (async).

### Reused as-is
`FillNa(fill)` and `Ffill()` already exist as causal, shape-preserving 1→1
functors. They are reused for aligned records (applied per component), not
reimplemented.

### Convenience parameters
`fillna=`/`dropna=` are offered as optional params on the **combinators only**
(`combine_latest`, `merge`, `resample`) — stream shaping is their job. They are
**not** added to the 100+ compute functors (the `"ignore"` policy already makes
those compute-correct; adding flags there would be surface explosion). Standalone
`fillna`/`ffill`/`dropna`/`filter` remain for explicit composition, so
`combine_latest(a, b, dropna="any")` and `dropna(combine_latest(a, b))` are two
spellings of the same result.

## Interaction with the existing API

- Bare-value calls are unchanged: `RollingMean(5)(array)` still returns a
  same-shape array; the implicit key is the row number.
- Existing multi-input functors (`RollingCorr(X, Y)`) keep their lockstep
  contract. Async/timed use becomes `RollingCorr(combine_latest(a, b))`.
- No compute functor is modified. All new surface lives in the combinator layer.

## Documentation

The principles above must be documented for users alongside the existing policy
pages (`docs/nan_policy.md`, `docs/polymorphic_api.md`, `docs/conventions.md`):

- A new page (working title `docs/multistream.md` / "Streams, keys, and
  alignment") covering: the order key and its two tiers (P1), the
  compute-vs-combinator / cardinality-preservation split (P2), alignment-as-a-
  separate-layer (P3), and the causality + cross-mode-identity guarantees (P4).
- Cross-link from `docs/polymorphic_api.md` (the lockstep case is the degenerate
  row-number key) and `docs/nan_policy.md` (fill vs drop; `ffill` == the
  `combine_latest` carry).

## Testing strategy

- **Cross-mode identity** — for every combinator, assert batch == streaming ==
  replay produce identical results (extends the existing
  `test_stream_vs_batch` / `test_stream_vs_generator` pattern).
- **Causality** — assert no output depends on future input; assert `bfill` and
  any lookahead op are absent by construction.
- **Combinator contracts** — `merge` ordering + tagging incl. tie handling;
  `combine_latest` firing rules (`when_all` vs `on_any`) and warmup; `dropna`
  `how` semantics; `resample` grid causality.
- **Degenerate case** — no-key combine equals today's lockstep behaviour
  bit-for-bit.

## Open decisions carried into planning

- Tie handling in `merge`/`combine_latest` when multiple sources share a key
  (proposed: deterministic by source order).
- Exact Python surface/naming of the tagged event from `merge` (source id form).
- Whether `resample` ships in this foundation spec or the next combinator spec
  (it is listed here for completeness; may be deferred to keep layer 1 tight).
