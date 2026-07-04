# Stream shaping in the DAG — Phase B: `resample`

**Status:** design approved 2026-07-04 (autonomous continuation; user reviews on wake)
**Scope:** A causal, windowed **`resample`** (downsampling) stream operation —
eager (+ `_iter`) and as a C++ DAG push-node. Two window modes (fixed
key-interval, fixed event-count) and a fixed reducer menu, all pure-C++
single-pass O(1) accumulators (no Python callbacks). Adds the engine's
**end-of-input flush** so a trailing partial bucket emits and batch == streaming
stays byte-identical.

This is **Phase B** of the stream-shaping effort; Phase A (`dropna` + `select`)
is merged. Design decisions here were settled with the user in conversation and
grounded in a cross-library research sweep (pandas, polars, kdb+/q, Flink,
Kafka Streams).

## Design principles (from the research)

- **The monotonic order-key IS the watermark.** No lookahead, no lateness
  machinery: a fixed key-interval bucket `[t0, t0+w)` is provably complete the
  instant an event with `key >= t0+w` arrives (keys never go backward). A fixed
  count bucket is complete on the Nth event.
- **Trailing partial bucket** stays un-emitted until an explicit end-of-input
  **flush** advances the frontier. Both batch and streaming flush at end of a
  finite feed, so both emit the trailing bucket → identical results. In a truly
  unbounded live stream the partial waits for completing events (correct causal
  behavior).
- **Boundaries:** default **left-closed `[t0, t0+w)`, left-label, floor-to-grid**
  — the cross-library consensus and the causally-safe choice. `label="right"` is
  allowed (display-only; the finance "bar stamped at close time" convention).
  Right-*closed* is **not** offered (it flirts with lookahead).
- **Single-pass O(1) accumulators**, NaN-ignore (matching the library's "ignore"
  policy). One accumulator computes everything; the reducer only selects fields.

## Scope decisions

- **Integer key-space.** The DAG engine is `int64`-keyed. `resample` buckets in
  integer key-space: `width` and `origin` are integers, bucketing uses integer
  floor-division. The eager form casts keys to `int64` so it matches the engine
  exactly (the identity oracle). Float-key resampling is a future extension.
- **Width-1 input.** Reducers consume a width-1 value stream. `agg="ohlc"`
  expands to width-4 output (`open/high/low/close`); all other reducers emit
  width-1. A wide (>1) input frame is a clear error (resample-then-combine is the
  documented multi-stream pattern). This keeps accumulators scalar-simple.
- **Sparse output.** Only buckets that received ≥1 event are emitted; empty
  interior buckets (gaps) are skipped (kdb/polars behavior), so no full grid is
  needed. A bucket with events but all-NaN values IS emitted (its reducers yield
  NaN / 0 per the NaN rules below).

## The operation

### API

Eager: `resample(keys, values, *, width=None, count=None, agg="last", origin=0, label="left")`
→ `(out_keys, out_values)`. Exactly one of `width` / `count` must be given (else
`ValueError`). `origin` applies to `width` mode only. `label ∈ {"left","right"}`.

`resample_iter(events, *, width=None, count=None, agg="last", origin=0, label="left")`
— streaming-generator twin over `(key, value)` tuples.

Graph: `resample(stream, *, width=/count=, agg=, origin=, label=)` where `stream`
is a `Node` (detected by `is_node`). Builds a graph node.

### Modes

- **by key-interval** (`width=W`, `origin=O`): bucket index
  `b = floordiv(key - O, W)`; bucket span `[O + b·W, O + (b+1)·W)`.
  `label="left"` stamps `O + b·W`; `label="right"` stamps `O + (b+1)·W`.
- **by event-count** (`count=N`): every `N` events form a bucket. `label="left"`
  stamps the first key in the bucket; `label="right"` stamps the last key.

### Reducers (`agg`)

One accumulator per open bucket holds: `count` (non-NaN count), `sum`, `min`,
`max`, `first`, `last`, and `has_event` (any event fell in the bucket — decides
emit-vs-skip). Each **non-NaN** value updates count/sum/min/max/first/last; NaN
values are ignored (but still set `has_event`). Emit maps `agg` → fields:

| agg | output | empty-of-finite (count==0) |
|---|---|---|
| `first` | first non-NaN | NaN |
| `last` | last non-NaN | NaN |
| `min` | min | NaN |
| `max` | max | NaN |
| `sum` | sum | 0.0 (pandas `min_count=0`) |
| `count` | count (as double) | 0.0 |
| `mean` | sum/count | NaN |
| `ohlc` | `[first, max, min, last]` (width-4) | `[NaN,NaN,NaN,NaN]` |

### Causal emit rule

- **by key-interval.** Track the current bucket index `b`, its accumulator, its
  label, `has_event`. On event `(k, v)`: `nb = floordiv(k - O, W)`. If first
  event ever → set `b=nb`, start bucket, add. If `nb == b` → add. If `nb > b` →
  **emit** the current bucket (if `has_event`), then jump to `b=nb` (intermediate
  empty buckets skipped), start bucket, add. On **flush** → emit current bucket
  if `has_event`.
- **by event-count.** Track counter, accumulator, first-key, last-key. On event →
  add, `++counter`; when `counter == N` → emit (label from first/last key), reset.
  On **flush** → if `counter > 0`, emit the partial bucket.

`floordiv(a, b)` is true floor division (handles `a < 0` correctly), `b > 0`.

## Architecture

### C++ (`include/screamer/dag/`)

**`ResampleNode<Key>`** (new `resample_node.h`) — a stateful `Sink<Key>` holding:
mode (`ByKey`/`ByCount`), `width`/`origin` or `count` (int64), `agg` (enum), `label`
(enum), the accumulator state, current-bucket bookkeeping, and `Sink& downstream`
+ a reused `out_` buffer (width 1 or 4). `push` runs the emit rule + accumulate;
`flush` emits the trailing bucket then forwards `downstream.flush()`; `reset`
clears all state (new for stateful nodes). Requires `f.width == 1` (throws
otherwise). Zero per-event heap allocation (fixed-size accumulator + reused
`out_`).

The accumulator is a tiny internal struct with `reset()`, `add(double)`,
`emit(Agg, double* out)`. One `add`, one `emit` — the reducer only selects.

### Engine flush (`include/screamer/dag/compiled_graph.h`, bindings, `dag.py`)

- `CompiledGraph::flush()` — `for (auto* s : input_sinks_) if (s) s->flush();`
  (the same call `run_batch` already makes at end-of-batch). Emits every open
  trailing bucket. Existing nodes emit nothing on flush (forward only), so this
  is backward-compatible.
- Bind `_CompiledGraph.flush()`.
- `Dag.stream` calls `self._cg.flush()` after the push loop, before `drain()`,
  so streaming mimics batch's end-of-input and stays byte-identical. `run_batch`
  already flushes — batch needs no change.
- `ResampleNode` is stateful → registered in a `reset_resamples_` list (mirroring
  `reset_combines_`) so `reset()` clears it between runs.

### Graph spec / builder / bindings / dispatch

- `NodeKind::Resample`; `NodeSpec` gains resample params (mode, width, origin,
  count, agg, label) — grouped in a small `ResampleParams` sub-struct to avoid
  bloating `NodeSpec` with six scalars.
- `GraphBuilder::add_resample(inputs, params)`; `PyGraphBuilder.add_resample`.
- `CompiledGraph`: node-width case (`Resample` → `agg==ohlc ? 4 : 1`); wiring case
  (single-input, stateful → registered for reset).
- `screamer/dag.py` dispatch: `"resample"` → `gb.add_resample(inp, ...)` (agg,
  label, mode as small enums/ints across the binding).
- `screamer/streams.py`: `resample` detects a `Node` first arg → `make_combinator_node`.

## Testing

- **Eager `resample`** (Task 1): each agg on a hand-worked series (both modes);
  label left/right; origin offset; NaN-ignore (all-NaN bucket, mixed); sparse
  gaps skipped; trailing partial emitted; ohlc width-4; `resample_iter` matches
  eager; width/count exclusivity + bad-agg + wide-input errors.
- **Graph identity** (Tasks 2 & 3): for every agg and both modes, a `Dag` with
  `resample(input, …)` gives **batch == stream == eager-oracle** — including the
  trailing-bucket flush (the batch/stream symmetry is the whole point), sparse
  gaps, `label="right"`, `origin`, ohlc, and `resample` composed with a
  downstream functor and with `combine_latest` (resample-then-combine).
- **Regression:** existing identity/stream tests still pass (flush is a no-op for
  non-resample graphs — assert an existing streaming dag is unchanged).

## Non-goals (future)

- Float-key-space resampling; wide (>1) input reduction per column.
- Overlapping/hopping windows (`period > stride`); session windows; quantile/median
  (needs bucket buffering); VWAP/weighted mean (needs a paired weight stream).
- `upsample` (increasing resolution). These are all separable later increments.
