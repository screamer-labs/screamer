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
the middle, a per-bar reducer at the root.

```python
price, volume = Input("price", price_arr), Input("volume", vol_arr)
bars = resample(index=t, every=BAR, agg={
    "open":     First()(price),
    "high":     ExpandingMax()(price),
    "low":      ExpandingMin()(price),
    "close":    Last()(price),
    "buy_vol":  ExpandingSum()(PosPart()(volume)),
    "sell_vol": ExpandingSum()(NegPart()(volume)),
})
# bars: a Stream, .values (N_bars, 6), .columns = the dict keys
```

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
  `Stream.columns`. Only non-empty bars emit; the trailing partial bar flushes **once**.
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

### Input binding (RECOMMENDED, confirm on review)

**Recommended: data-carrying `Input`.** Extend `Input` to optionally hold its stream:
`Input("price", price_arr)`. Then the lazy expressions have their data, and the
`resample(agg={...}, index=t, every=BAR)` call needs no separate data plumbing. Pure
`Input("price")` (no data) keeps working for reusable graphs. Alternatives, if
preferred:
- explicit map: `resample(agg={...}, index=t, inputs={"price": arr, "volume": arr})`;
- kwargs: `resample(agg={...}, index=t, price=arr, volume=arr)` (concise but can
  collide with resample's own parameter names).

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

- Input-binding style (recommend data-carrying `Input`).
- Lazy-dict vs current single-input dict: coexist (recommended) or replace.
- Whether `First`/`Last` also matter outside bars (they are already resample agg
  strings; the functors are new, needed for the lazy form).

## Rough phases

1. **Fix `combine_latest` flush coalescing** + regression test (independent, ships on
   its own).
2. **`First` / `Last`** reducer functors (add-a-function checklist).
3. **Multi-column resample node** (C++): one clock, N reducer sub-graphs, one aligned
   labelled row per bar. Generalize `GenericResampleNode`.
4. **Lazy-dict front-end** + input binding; compile the dict to the node.
5. **Docs + a notebook** using the lazy form (replacing the two-call recipe in
   `12-custom-bars-with-agg-dict` with the one-call lazy version).

All numeric logic in C++ (the node, the reducers); Python stays a thin marshalling /
compile shim, so pure-C++ and a future WASM binding get the full behavior.
