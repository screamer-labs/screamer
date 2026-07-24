# Event-time watermarks: correct live Delay, merge, and resample

**Status:** design, pending review
**Date:** 2026-07-24
**Supersedes the limitation in:** `docs/functions_streams/Delay.md`, the follow-on note in `docs/superpowers/plans/2026-07-22-delay-and-forecast-pairs.md`

## Problem

In a live compiled `Pipeline`, a `Delay(d)` feeding a `CombineLatest` merge produces
wrong output. `push_event(input, index, value)` propagates synchronously through the
whole subgraph in one call stack. `DelayNode` re-stamps a frame's index `t -> t+d` and
pushes it downstream immediately, so the merge receives a future-dated frame while its
sibling ports are still at `t`. The merge aligns the delayed event against stale
as-of values.

Batch does not have this problem: `MergeSource` (a min-heap over the input sources)
feeds the graph in global index order, so when a delayed frame at `t+d` reaches the
merge, every sibling frame with index `<= t+d` has already been applied.

`batch == live` is a hard invariant in screamer (the causality rule: stored-data and
streaming paths must give identical results). Batch feeds the merge in global index
order, so **live must too**. The current live path does not, which is the defect.

## Core idea

Introduce an **event-time watermark** as a first-class signal in the DAG. A watermark
`W` on an edge is a promise: every future frame on that edge has index `>= W`. It is
monotone non-decreasing and flows the same topology as data.

The merge holds incoming frames in a bounded reorder buffer and releases them into its
existing as-of logic in global index order, gated by the minimum per-port watermark.
`Delay` shifts the watermark by `+duration` (mirroring what it does to the data index).
`Filter` / `DropNa` forward the watermark past frames they drop. `Resample` closes
windows on the watermark, retiring its separate `advance()` path. Live thereby
reproduces batch's global-order delivery: `batch == live` holds by construction, and
the fix composes through the whole operator set rather than being a merge-only patch.

### Why an explicit watermark, not inferred progress

A merge could try to infer each port's progress from the last data-frame index it saw.
That breaks in a composed pipeline: a `Filter` or `DropNa` between an input and the
merge drops data frames, so the inferred watermark stalls and the merge hangs, never
releasing the delayed sibling. An explicit watermark survives the drop (the dropping
node forwards it even when it swallows the data). The explicit model also subsumes the
existing `advance()` special-case and unblocks genuinely idle ports. The inferred
approach is not merely less general; it is incorrect for pipelines with a dropping node
upstream of a delayed merge.

### The watermark drives a buffer, it does not replace it

A future-dated frame stamped `t+d` still has to be held until it is safe. The merge
keeps a reorder buffer. The watermark is the correct, composable **release trigger** for
that buffer. `Delay` stays a stateless transform (shift data index and watermark by
`+d`); the buffering lives at the merge, which is the only node that sees all ports.

## The watermark contract

- **Meaning.** A watermark `W` delivered on an edge guarantees no future frame on that
  edge has index `< W`. Monotone non-decreasing per edge.
- **Sources.**
  1. A data frame at index `i` implies watermark `>= i` on that edge (inputs are
     per-input sorted). Normal data flow carries watermarks for free.
  2. `advance(now)` injects a watermark on all inputs without a data frame (idle
     advancement; the generalization of today's Resample-only `advance()`).
  3. `flush()` is watermark `= +INF`.
- **Transformation.** Each node forwards or transforms watermarks (`Delay` shifts,
  the merge gates, `Resample` closes windows). Most nodes forward unchanged.

## Sink protocol change

Add to `Sink<Index>`:

```cpp
// No future frame delivered to this sink will have index < w. Monotone.
// Default: forward to the single downstream sink (nodes with a downstream_).
// Terminal/output sinks consume it; gating nodes (CombineLatest) override.
virtual void on_watermark(Index w);
```

A node's per-edge watermark is `max(highest data-frame index seen, highest
on_watermark seen)`. So a data frame advances the watermark implicitly; explicit
`on_watermark` is needed only where data does not flow (drops, idle, flush). This keeps
the common path allocation- and call-free: no extra watermark call per data frame.

## Per-node behavior

- **Pass-through nodes** (`Functor`, `Select`, broadcast fan-out): forward
  `on_watermark(w)` to the downstream sink(s) unchanged; data frames keep their index.
  Broadcast forwards the watermark to every consumer (so `x` feeding both a direct
  merge port and a `Delay` propagates correctly to both).
- **`Delay(d)`**: on a data frame, emit `{index + d, values}` (unchanged); on
  `on_watermark(w)`, emit `on_watermark(w + d)`. Stays stateless, no reset state.
- **`Filter`**: on a kept frame, forward it (its index implies the watermark); on a
  dropped frame at index `i`, forward `on_watermark(i)` so downstream time still
  advances; forward explicit `on_watermark(w)`.
- **`DropNa`**: same rule as `Filter` for dropped (all-/any-NaN) rows.
- **`CombineLatest` (the merge):** the one node with real new state.
  - Track a per-port watermark `wm[j]`.
  - Buffer incoming frames whose index exceeds the current `low_wm = min_j wm[j]`.
  - On any watermark or frame that advances `low_wm`, release buffered frames with
    index `<= low_wm` in nondecreasing index order into the existing `on_port()` as-of
    and coalescing logic. Because release is in global index order, by the time a
    delayed frame at `t+d` is released, every frame with index `< t+d` has been applied,
    so `cl_.latest()` is the correct as-of state.
  - Forward `on_watermark(low_wm)` downstream so a node after the merge (for example a
    `Delay` after a `CombineLatest`) does not stall during coalescing gaps.
  - Composition with `when_all`: `when_all` still governs whether `cl_` fires before all
    ports are seen. A frame can be watermark-safe yet not emit because `when_all` is
    still waiting; that matches batch, which feeds the same `cl_` in order.
  - Flush is the ultimate watermark: `on_watermark(+INF)` on every port drives
    `low_wm = +INF`, drains all buffered frames in order, emits the final coalesced row
    once, then forwards `flush()` downstream. The existing per-port flush bitmask is
    reframed as "every port reached `+INF`."
- **`Resample`**: close each window when the incoming watermark passes its boundary,
  instead of on a separate `advance()` call. `advance(now)` becomes `on_watermark(now)`
  injected at the inputs and flowing to the resample node like any other watermark. Data
  frames continue to close windows as their indices pass boundaries (as today); explicit
  watermarks close windows for idle input.

## User-facing API (`_LiveDag`)

Unchanged in surface, generalized in meaning:

- `push_event(input, index, value)`: pushes a data frame; implies watermark `index` on
  that input.
- `advance(now)`: injects `on_watermark(now)` at every input; now flows through the
  entire graph (previously only reached resample nodes). This is the escape hatch to
  release a delayed buffer that is waiting on an idle stream.
- `flush()`: injects `on_watermark(+INF)` at every input; drains reorder buffers, closes
  all windows, emits final rows, forwards flush.
- `result()` / `drain()`: unchanged.

No change to the frozen public operator API. Watermarks are an internal engine
mechanism.

## Overflow and bounds

The merge's reorder buffer holds frames with index in `(low_wm, leading_edge]`. For the
`Delay(d)` pattern its footprint is one delay-window of events, roughly
`max_delay * event_rate`, plus any lag from an idle port.

- A configurable per-merge cap `max_pending` bounds it.
- On exceed, raise a clear error (for example: "CombineLatest reorder buffer overflow:
  input has not advanced past index X; a stream is stalled, call advance(now) or check
  the feed"). Never silently drop and never grow unbounded. This matches screamer's
  fail-loud ethos.
- The remedy for a legitimately idle stream is `advance(now)`, which advances the
  watermark and releases the buffer.

Default cap: a generous constant (proposed 1_000_000 frames), overridable at
`Pipeline` construction. The plan will pick the exact wiring.

## Batch path and `batch == live`

Batch keeps `MergeSource` for input ordering (correct and the hot path; unchanged). The
merge node is watermark-aware in both modes. In batch, frames arrive already in global
index order, so the reorder buffer is a passthrough (each frame's index immediately
`<= low_wm`) and behavior is identical to today. Batch `flush` drains normally.

`batch == live` becomes a directly tested invariant: the release rule "emit buffered
frames with index `<= min_j wm[j]`" is exactly `MergeSource`'s safety condition ("a
frame at index `e` is safe once no port can still produce index `< e`"). Correct
watermarks imply live delivery order equals global index order equals batch order;
`flush` drains the tail. Equivalence is asserted across the Delay/merge topologies and
the Filter/DropNa/Resample compositions.

## Edge cases

- **Chained or multiple delays into one merge**: per-port watermarks and the shift
  compose; each delayed port releases against the min of the others.
- **`Filter`/`DropNa` upstream of a delayed merge port**: watermark forwarding on drop
  prevents a stall (a dedicated test).
- **Idle port**: `advance(now)` releases a waiting buffer without a data frame (a
  dedicated test).
- **`Delay` after a merge / nested pipelines**: `Delay` shifts the merge's forwarded
  watermark; no special handling.
- **Negative / lookahead delay**: out of scope; a backward index shift violates
  causality and is rejected (positive duration only, as today).
- **Multi-output lazy driver**: the `LazyDriver`'s existing output-side watermark is a
  separate, output-alignment concern; the new within-graph watermark must not conflict
  with it. Verified by the multi-output tests.

## Testing strategy

- Per-node watermark unit tests: `Delay` shift, `Filter`/`DropNa` forward-on-drop,
  `Resample` close-on-watermark, pass-through forwarding.
- The definitive case: `Delay -> CombineLatest` fused in a live `Pipeline`, asserting
  `batch == live` identical output across several durations and irregular feeds (the
  currently-broken case, now correct).
- `Filter`/`DropNa` between input and a delayed merge port: no stall, correct output.
- Idle port plus `advance(now)`: releases correctly.
- Resample via watermark equals the old `advance()` behavior: existing resample tests
  pass unchanged.
- Overflow cap raises loudly.
- Full regression: `test_streams_combine_latest`, `test_streams_delay`,
  `test_dag_delay`, `test_streams_resample`, `test_supervised_forecast_pairs`, and the
  whole suite still pass.

## Out of scope

- Replacing the batch `MergeSource`.
- Any change to the frozen public operator API.
- Negative/lookahead delay.
- New user-facing time semantics beyond generalizing `advance(now)`.

## Files touched (anticipated)

- `include/screamer/dag/frame.h` (Sink gains `on_watermark`).
- `include/screamer/dag/delay_node.h` (shift watermark).
- `include/screamer/dag/combine_latest_node.h` (reorder buffer + per-port watermark +
  release + overflow cap).
- The `Filter`, `DropNa`, `Select`, and pass-through/broadcast node headers (forward
  watermark; forward-on-drop where applicable).
- `include/screamer/dag/resample_node.h`, `resample_generic_node.h` (close on
  watermark; unify `advance`).
- `include/screamer/dag/compiled_graph.h` (`advance(now)` injects a watermark at inputs;
  `flush` injects `+INF`; wiring).
- `screamer/dag.py` / bindings (`advance`/`flush` semantics; `max_pending` config).
- `docs/functions_streams/Delay.md` (remove the limitation; document the corrected
  behavior).
