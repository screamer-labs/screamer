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
- **Arbitrary non-numeric key types (string/tuple orderables).** The combinator
  layer is implemented in **C++** for speed and to be reusable as C++ DAG nodes
  later (see "C++ architecture"). Comparing arbitrary Python objects in C++ is
  slow and defeats that purpose, so keys are **numeric only** — see the amended
  P1. A Python-object key path is explicitly out of scope.

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
  numbers and any numeric key. Enables merging, alignment, and backtest
  (max-speed) replay.
- **Metric key** (differences are *meaningful* — `datetime64`, `int64` ns, float
  seconds) — additionally enables **wall-clock replay**, because a sleep
  duration only exists when key-deltas convert to time.

**Amendment (C++ decision): keys are numeric only.** Because combinators are
implemented in C++ (see "C++ architecture"), the key is carried as a numeric C++
type — `int64_t` **or** `double`, chosen at the Python boundary from the input
dtype. This covers every realistic timestamp: `datetime64[*]` (an `int64` view,
lossless), integer keys, `float64` seconds, pandas `DatetimeIndex`, native
`int`/`float`, and the row-number fallback. It **excludes** non-numeric
orderables (string/tuple keys). A single graph/`merge` uses one key type across
all its sources (you cannot meaningfully compare `datetime64[ns]` with float
seconds anyway). `double` keys carry `datetime64[ns]` unsafely (ns-since-epoch
exceeds 2^53), so `datetime64`/`int64` inputs always take the `int64_t` path.

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

## C++ architecture: pull sources driving a push graph

Combinators are C++ so they run at speed and so the phase-3 DAG can be an
all-C++ structure (a node whose sink is another node — "a function of a
function") with no per-event round-trip through Python. Every combinator and
every compute functor presents a common node interface, in one of two roles:

- **Source layer = pull.** `template<class Key> struct Source { virtual
  std::optional<Event<Key>> next() = 0; };`. `merge` is a k-way heap over N
  sorted `Source`s emitting tagged events; `pace` wraps a `Source`. The source
  produces one **totally-ordered** event sequence.
- **Graph interior = push.** `template<class Key> struct Sink { virtual void
  push(const Event<Key>&) = 0; virtual void flush() {} };`. Interior nodes hold
  their downstream sink(s) and emit into them: compute-functor adapters,
  `combine_latest`, `filter`, `dropna`, `split`.
- **Driver.** `while (auto e = source.next()) graph.push(*e);` then
  `graph.flush()`. Sinks collect outputs.

`Event<Key> { Key key; double value; uint32_t source; }` — value is always
`double` (screamer invariant); `source` tags provenance for `merge`. The node
model is templated on `Key ∈ {int64_t, double}`, instantiated for both; the
Python boundary picks the instantiation from the input key dtype.

**Why this shape:**
- **The DAG is this graph.** Phase 3 "materialize" = construct the node objects
  and wire the sinks. No new execution model is invented later.
- **Cross-mode identity becomes structural.** Batch, streaming, and replay are
  three *drivers* feeding the *same* graph — they cannot diverge numerically,
  because only the source differs. Pacing lives entirely in the source, so it
  changes *when* events flow, never their values.
- **The existing functors are reused unchanged.** A thin adapter wraps a
  `ScreamerBase` (or `FunctorBase<_,N,1>`) as a push node: on push it computes
  and emits, key passing through (shape-preserving, P2).
- **`combine_latest` fuses alignment with its N-input consumer.** It holds
  `latest[N]`/`seen[N]`; on any port push it updates, and when warm (or on
  `emit="on_any"`) it feeds the aligned `std::array<double,N>` to a downstream
  `FunctorBase<_,N,1>` node, emitting a single-valued event. This is exactly how
  `RollingCorr(combine_latest(a, b))` is realized in C++.

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

### `resample(source, clock)` — deferred to the combinator spec
Aligning a stream to a reference clock/grid is cardinality-changing and causal
(emits a grid point from data at keys `≤` that grid point). It is **not** part of
this foundation; it lands in the next (combinator) spec. Listed here only so the
layer boundary is clear. The true minimum for layer 1 is
`merge` + `combine_latest` + `filter`/`dropna` + `split` + `pace`.

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

## Implementation-plan decomposition

The foundation is large enough to build as **four sequential plans**, each a
working, testable increment building on the prior's interfaces:

1. **C++ event/node core** — `Event`/`Source`/`Sink` (templated on `Key`), the
   batch driver, the compute-functor push adapter, Python bindings + carrier
   detection for the single-stream case. Deliverable: `source → functor →
   collector` graph whose batch output equals the existing functor bit-for-bit.
2. **Source/driver layer** — `merge` (k-way heap, tagged) and `pace` (backtest +
   wall-clock replay, async).
3. **Interior combinators** — `combine_latest` (fused N-input, firing rules),
   `filter`/`dropna`, `split`.
4. **Docs + cross-mode-identity hardening** — the `docs/multistream.md` page,
   cross-links, and the batch==stream==replay test matrix across all combinators.

## Open decisions carried into planning

- Tie handling in `merge`/`combine_latest` when multiple sources share a key
  (proposed: deterministic by source order).
- Exact Python surface/naming of the tagged event from `merge` (source id form).
