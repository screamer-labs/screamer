# Multi-column bars, lazy `agg` expressions, and `combine_latest` flush coalescing

**Status:** design (pre-implementation)
**Date:** 2026-07-08
**Follows:** `2026-07-08-windowed-aggregation-design.md` (the `resample(agg=str|functor|dict)` work, now merged)

## Goal

Let each bar column name its own input and reducer through a **lazy expression**, so
a full OHLCV2 (or any custom multi-input bar) is one declarative call. Back it with a
single **"collect and reset" multi-column resample node** so columns are aligned by
construction. Along the way, fix a `combine_latest` end-of-input bug found while
exploring the DAG pattern.

## Exploration findings (verified, they drive this design)

- `functor()(Node)` composes **lazily** (a graph node), unlike `functor()(array)`
  which evaluates eagerly. Confirmed: `ExpandingMax()(Input("price"))` and
  `ExpandingSum()(PosPart()(Input("price")))` are both lazy nodes.
- `resample(node, every=BAR, agg=functor)` already reduces per bar (Task 4). A single
  graph resample yields the correct bars.
- Building OHLCV2 as **N per-column resample nodes glued with `combine_latest`**
  MIS-aligns: the graph output index came out `[0 40 80 120 160 160]` - the final bar
  label is **duplicated**. Eager `combine_latest` on two identical-index streams is
  correct (5 rows). So the fault is a graph, end-of-input issue.

Two separable work items follow: a `combine_latest` bug fix, and the new multi-column
bar node.

---

## Part A: `combine_latest` flush coalescing (bug fix)

### Contract (the sensible expectation to test against)

`combine_latest(s1, ..., sN)`:

- emits **exactly one row per distinct index** across the inputs;
- each row at index `k` carries every input's most-recent value as-of `k`
  (latest-value-forward);
- **same-index events coalesce into one row, whether they arrive mid-stream or at
  end-of-input flush**; the trailing flush emits the final coalesced row **once**;
- `emit="when_all"` starts rows once every input is warm; `emit="on_any"` from the
  first event.

### The bug

The graph `combine_latest` node coalesces same-index events during streaming (labels
0-120 are single) but at **end-of-input flush** each input's flush of its final
(same-index) event triggers a *separate* emit, producing a duplicate final row
(`[... 160 160]`). Likely in `include/screamer/dag/combine_latest_node.h` `flush()`.

### Fix + test

Coalesce by index at flush, mirroring the mid-stream path (emit the final row once).
Regression test: N streams sharing indices (including the final one) must yield
exactly `len(distinct indices)` rows, no duplicate final index, values equal to the
eager `combine_latest`. This is worth fixing independently: it affects any graph that
combines streams, not only bars.

---

## Part B: the multi-column "collect and reset" bar node

One node, one bar clock, N columns. Alignment is structural (no `combine_latest`), so
it sidesteps Part A entirely.

### Front-end: lazy `agg` dict

Each dict value is a **lazy expression**: an input at the leaf, per-tick transforms in
the middle, a per-bar reducer at the root. Inputs are **pure symbolic placeholders**;
the bar aggregator is a **node authored via the dict** and placed in a `Dag`. Data
(batch or stream) is bound **at call time**, not at input construction (see "Input
binding" and [[project_dag_is_primitive]]).

```python
price, volume, t = Input("price"), Input("volume"), Input("t")
bars_node = bars(index=t, every=BAR, agg={
    "open":     First()(price),
    "high":     ExpandingMax()(price),
    "low":      ExpandingMin()(price),
    "close":    Last()(price),
    "buy_vol":  ExpandingSum()(PosPart()(volume)),
    "sell_vol": ExpandingSum()(NegPart()(volume)),
})
dag = Dag(inputs=[t, price, volume], outputs=[bars_node])
bars = dag(t=t_arr, price=price_arr, volume=vol_arr)   # bind data, batch OR stream
# bars: a Stream, .values (N_bars, 6), .columns = the dict keys
```

The eager `resample(...)` stays single-input sugar; the multi-column, multi-input case
is expressed as a `Dag` with a bar node. `bars(...)` is the node constructor that
compiles the dict into the multi-column "collect and reset" node (Part B), so it can be
placed anywhere in a graph and reused across datasets.

### Node contract

- **Inputs:** one bucketing spec (`every` xor `count`, `origin`, `label`) over a shared
  index; and N columns, each a lazy sub-graph rooted at some input.
- **Per bar:** feed the bar's samples through each column's sub-graph; at bar close
  record each column's output, then **reset every stateful node in each column's
  sub-graph**. (No need to distinguish "reducer" from "transform": resetting the whole
  sub-graph per bar is the general rule. `PosPart` is stateless so its reset is a
  no-op; `ExpandingSum` resets to restart the bar.)
- **NaN-ignore per column:** skip NaN samples; a column with no finite sample in a bar
  emits NaN.
- **Emit:** exactly one row per bar with all N columns, in dict order, labelled via
  `Stream.columns`. A bar is finalized when its window closes - triggered by an event
  crossing the boundary, by a **time/clock advance**, or by an explicit **flush** (see
  Part C). Empty bars are handled per the `fill` policy (Part C); the trailing partial
  bar flushes **once**.
- **Causal:** emit at bar close only. `batch == streaming == oracle`.
- **Guarantee:** one clock means columns cannot drift; the row for bar `k` is
  `[col1_k, ..., colN_k]` by construction.

This is the "collect (accumulate per column within the bar) and reset (at each
boundary)" function.

### New functors: `First`, `Last`

`ExpandingMax/Min/Sum/Mean` already exist. The literal OHLC example needs two more
1->1 reducers: `First` (latch the bar's first value) and `Last` (the bar's last value;
`Identity` already behaves as last under per-bar reset, but a named `Last` reads
better). Both stateless-to-reset, NaN policy `ignore`.

### Input binding (RESOLVED: define-then-bind, the Dag is the primitive)

`Input(name)` stays a **pure, data-less symbolic placeholder** - no data-carrying
`Input`. The model is **define-then-bind** (the existing `Dag` contract, see
[[project_dag_is_primitive]]):

1. **Define:** author the lazy dict into a `bars(...)` node, then wrap it in a
   `Dag(inputs=[...placeholders...], outputs=[bars_node])`. The `Dag` acts like a
   reusable functor with named inputs and outputs.
2. **Bind at call time:** `dag(t=t_arr, price=price_arr, volume=vol_arr)` by Input name
   (or positionally, in `inputs` order). The same `Dag` runs a NumPy array (batch) or a
   live stream, with `batch == stream`.

No new binding plumbing is needed: `Dag.__call__` already binds by name / position and
`get_input_nodes` already discovers the reachable `Input`s. The bar node's per-column
sub-graphs are ordinary graph nodes rooted at those placeholders, so the existing
compile + bind path carries them.

---

## Part C: time-driven and on-demand flush (cross-cutting)

Today's bucketing is purely **event-driven**: a window closes only when the next event
crosses its boundary, or at end-of-input. So with sparse trades, bars emit late and
empty windows vanish. That is wrong for a continuous minute-bar series, and finalizing
on demand is needed across the streaming operators, not just bars.

Two finalization triggers, in addition to "an event crossed the boundary":

- **On-demand flush** (`flush()`): finalize the current partial window now, e.g. at the
  end of a processing loop. End-of-input flush is the special case at stream end.
- **Time-driven finalization** (`advance(now)`): close every window whose boundary has
  passed by logical time `now`, even if no event fell in it. With minute buckets, a
  clock tick at the minute boundary closes that minute's bar with zero trades.

### Empty-window policy (`fill`)

When a window closes with no finite samples, its output depends on a `fill` policy:

- `skip` - no row (today's behavior).
- `nan` - an all-NaN row at the window label.
- `carry` - carry forward (for OHLC: open=high=low=close=previous close; sums/volume=0).
  The usual choice for a continuous minute-bar series.

`fill` is per operator; for multi-column bars it may later be per column (OHLC carries,
volume zeros). Recommend default `skip` to preserve current behavior, `carry`/`nan`
opt-in.

### Mechanism

A clock is just another indexed, **value-less event** (a "heartbeat" / watermark) in
screamer's stream model: any index advance, from a trade *or* a heartbeat, moves logical
time forward and closes completed windows. So `advance(now)` is sugar for feeding a
heartbeat at `now`. Provide both: feed heartbeats in a stream, or call `advance(now)` /
`flush()` directly on the operator or the `Dag`. This needs a value-less event in the
C++ `Frame` (a width-0 / punctuation event) so a heartbeat advances time without
contributing a sample to any reducer.

### Cross-cutting protocol

`flush()` and `advance(t)` belong on the streaming interface generally - the multi-column
bar node, `resample`, `combine_latest`, streaming stateful functors, and the `Dag` -
via a common `Flushable` / `advance` protocol. The batch path finalizes at end-of-input
as today; the live path gains explicit, clock-driven control. (This also gives
`combine_latest` and friends a principled place to finalize, which ties back to Part A.)

### Open questions

- `fill` default (recommend `skip`; `carry`/`nan` opt-in).
- Heartbeat-event vs explicit `advance(now)` call - recommend supporting both, with
  `advance` as sugar over a heartbeat.
- Scope of this pass: land `flush` + `advance` + `fill` on the resample / bar path first,
  then extend the protocol to the other streaming operators - or do the whole protocol at
  once (larger).

## Relationship to existing work

- This **supersedes** the deferred follow-up "consolidate dict/`ohlcv` per-column
  recompile into one shared-`Input` `Dag`". This node *is* that consolidation, done
  correctly (alignment guarantee, single pass).
- Reuses the DAG lazy composition and generalizes Task 4's `GenericResampleNode`
  (single functor reducer) to multi-column / multi-input.
- The existing `agg=str` and `agg=functor` forms stay. The lazy-dict is a new, more
  expressive dict form. Open question: does the lazy-dict **replace** or **coexist**
  with the current `agg={name: reducer}` single-input dict? Recommend coexist
  (single-input dict stays the simple case; lazy-dict is the multi-input case).

## Open decisions (confirm before the plan)

- Input-binding style: RESOLVED - define-then-bind on a `Dag`, pure placeholder
  `Input`, no data-carrying `Input`. The multi-column bars is a node placed in a `Dag`;
  `resample` stays single-input eager sugar. (See "Input binding".)
- Lazy-dict vs current single-input dict: coexist (recommended) or replace.
- Whether `First`/`Last` also matter outside bars (they are already resample agg
  strings; the functors are new, needed for the lazy form).

## Rough phases

1. **Fix `combine_latest` flush coalescing** + regression test (independent, ships on
   its own).
2. **Value-less heartbeat event + `flush()` / `advance(now)`** on the resample path,
   plus the `fill` policy (`skip`/`nan`/`carry`) for empty windows (Part C). This is the
   foundational streaming capability; land it on `resample` first with tests
   (time-driven close of empty minute buckets; manual flush at loop end).
3. **`First` / `Last`** reducer functors (add-a-function checklist).
4. **Multi-column resample node** (C++): one clock, N reducer sub-graphs, one aligned
   labelled row per bar; honors `flush`/`advance`/`fill` from phase 2. Generalize
   `GenericResampleNode`.
5. **Lazy-dict front-end** + input binding; compile the dict to the node.
6. **Extend `flush`/`advance` across the streaming protocol** (`combine_latest`,
   streaming functors, `Dag`) if not already, per the Part C scope decision.
7. **Docs + a notebook** using the lazy form and a clock-driven minute-bar example
   (replacing the two-call recipe in `12-custom-bars-with-agg-dict`).

All numeric logic in C++ (the node, the reducers); Python stays a thin marshalling /
compile shim, so pure-C++ and a future WASM binding get the full behavior.
