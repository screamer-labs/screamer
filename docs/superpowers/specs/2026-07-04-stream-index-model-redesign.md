# Stream / index model redesign (finalized)

**Status:** finalized (pre-release; no back-compat required)
**Scope:** Redesign the multi-stream *interface and the `combine_latest`
alignment semantics* so the model is honest: a stream is a sequence of
**values**; an **index** is an optional ordering coordinate (not a key, not a
dict); alignment produces **one row per distinct key**. Introduce a `Stream`
type, make combinators polymorphic on input type (raw arrays / `Stream` / graph
`Node`) with a mirrored return, rename `key(s)` -> `index` everywhere
user-facing, and default a missing index to position with no allocation. The
causal/identity guarantees are preserved and in fact strengthened (batch ==
stream == graph now holds for the aligning op, which it did not before).

## Motivation

The current docs and API call a stream "a sequence of `(key, value)` events" and
every combinator takes and returns `(keys, values)` pairs with the key mandatory
and first. Three things are wrong:

- **A stream is values.** The coordinate is *when/where* each value sits, not
  part of the payload, and it is optional (single-series functors need none).
- **"key" imports dict semantics.** There is no lookup, keys can repeat, the only
  operations are order and compare. It is an ordering axis.
- **`combine_latest` counted per event, not per key.** For two events at the same
  key it emitted a phantom intermediate row (one stream updated, the other stale
  at the same timestamp), which also made eager and graph disagree on row count.

Pre-release is the time to fix all three.

## Vocabulary

One word per concept, used consistently in code, docstrings, and docs. This table
is the enforceable reference; the banned alternatives must not appear on the
public surface (and, for `key`, not internally either).

| Concept | The word | Do not use |
|---|---|---|
| the payload datum / array | **value** / **values** | data, sample-as-payload |
| the ordering coordinate | **index** | **key**, order key, timestamp-as-name |
| a sequence of values + optional index (and the type) | **stream** / `Stream` | series, feed, signal |
| one `(value, index)` datum arriving live | **event** | tick, message, sample |
| one line of a 2-D batch result | **row** | - |
| a stream with `index=None` | **positional** | keyless, unindexed |
| a streams-layer operation | **stream operator** | **combinator** |
| a C++ compute op (RollingMean, ...) | **functor** | - |

Rules:
- **`key` is retired entirely**, including the internal C++ `Key` template and
  `key` variables, which are renamed to `Index` / `index`. A contributor should
  never meet "key" and inherit the dict connotation.
- **"combinator" is retired** for **stream operator**, in prose and in code
  identifiers (`make_combinator_node` -> `make_operator_node`; the `"combinator"`
  op-tag -> `"operator"`; etc.). Single-stream shape operators do not combine.
- **"event" and "row" are both kept, used precisely:** "event" only in the
  streaming / `_iter` / `pace` context; "row" only for a line of a 2-D result.
  No cross-drift.
- **"stream" is the single concept word.** Retire "series" (the `*series` params
  and `_normalize_series` become `*streams` / `_normalize_streams`); avoid "feed"
  as a concept name.

## The model

- A **stream** is `values` (1-D or 2-D numpy array), optionally annotated with an
  **index** (1-D array, same length): timestamps, a tick/message counter, or any
  orderable coordinate. **`index=None` means position** (row number / arrival
  order) and allocates nothing.
- **Values are primary; the index is optional metadata**, and the coordinate is
  called **`index`** everywhere (array, keyword, `Stream.index`, and the internal
  C++ type, now `Index` rather than `Key`). "key" does not appear anywhere.
- **No index means "the clocks are aligned."** Two no-index arrays are assumed to
  tick together (position i of one pairs with position i of the other). To model
  streams on *different* clocks (async, or a backtest of async feeds) you
  **provide an index**; without one, aligned clocks are assumed.

## Polymorphic in / out contract

Combinators dispatch on input type and mirror it on return - the same contract
the single-series functors already use (scalar/array/iterator/`Node`):

| You pass | You get back |
|---|---|
| raw value array(s), optional `index=` | `(values, index)` - 2-tuple; `index is None` when positional |
| `Stream` object(s) | a `Stream` |
| graph `Node`(s) | a `Node` (builds the DAG) |

- **One core.** Inputs normalize to `Stream` internally; the core runs once; the
  result is adapted back to the input regime by a single shared helper.
- **Raw return is always `(values, index)`** (values first), 2-tuple regardless of
  data, with `index=None` for positional so nothing is allocated. `vals, idx =
  combine_latest(a, b)` always works; `idx is None` is a checkable "no real
  ordering here."
- **Mixed regime:** a bare array auto-wraps to a positional `Stream`; the output
  is a `Stream` if any input was a `Stream`, else raw; a `Node` anywhere routes to
  the graph builder.

## `combine_latest` semantics: one row per distinct key

`combine_latest` emits **one row per distinct key**, holding each stream's latest
value at that key. Same-key updates across streams **coalesce** into a single
row (the settled joint state); genuine as-of rows at distinct keys survive. This
is the core semantic change.

### Key source by input type

| Input | Key source | Alignment | Output cardinality |
|---|---|---|---|
| no index (batch arrays) | row position (aligned clocks assumed) | lockstep; **equal length required** | N |
| explicit index (batch / `Stream`) | your timestamps | as-of, ties coalesced | # distinct keys |
| live async, keyless | arrival order (stamped on receipt) | emit per arrival (reactive) | # arrivals |
| live async, timestamped | your timestamps | as-of, coalesced | # distinct keys |

Worked example, `a@[1,2,4]=[10,20,40]`, `b@[1,3,4]=[1,3,4]`, `emit="when_all"`:

| key | a as-of | b as-of | row |
|---|---|---|---|
| 1 | 10 | 1 | (10, 1) |
| 2 | 20 | 1 | (20, 1) - b legitimately hasn't moved |
| 3 | 20 | 3 | (20, 3) |
| 4 | 40 | 4 | (40, 4) - both moved at key 4, coalesced to one row |

-> 4 rows (the old code returned 5, with a phantom second row at key 4).

### Rules and edge cases

- **No index requires equal length.** Aligned clocks tick together, so two
  no-index arrays must be the same length; unequal is an error ("streams have no
  index, so they are assumed aligned - lengths must match, or provide an index to
  align different clocks"), not a silent forward-fill.
- **`emit`**: `"when_all"` (default) drops leading keys until every stream is
  warm; `"on_any"` emits from the first key (NaN for not-yet-seen streams).
- **Live keyless async**: the streaming layer stamps a monotonic **arrival key**
  on each event (no shared clock exists otherwise), so every event has a distinct
  key and `combine_latest` emits one row per arrival - classic reactive behavior,
  correct for feeds at different rates (per-count pairing would stall a fast feed
  waiting for a slow one). Default arrival key is a counter (deterministic order);
  wall-clock is used when `pace`/timing needs it.
- **Reproducibility / backtest**: to reproduce a live keyless run or get batch ==
  stream identity, capture the arrival keys (or your own timestamps) at ingestion
  and replay that keyed stream in batch. You cannot reconstruct the interleaving
  from two static keyless arrays afterward - which is exactly why the coordinate
  is metadata that must be captured live.

### Why coalescing unifies the modes (identity)

The reason eager and graph disagreed (3 vs 5 rows for positional) was the missing
coalesce. The C++ core already emits in key order, so **the last row of each
key-run is the settled joint state**; coalescing == "keep the last row of each
key-run." Applied in every mode, the modes converge:

- eager positional as-of `[0,1,1,2,2]` -> `[0,1,2]` = the lockstep 3 rows
  (`np.column_stack` is a valid shortcut for the equal-length positional case).
- eager indexed -> distinct-key rows.
- graph batch -> collapse the collected output the same way -> matches eager.
- graph/live streaming -> the combine node emits a key's row only when the key
  advances (an event with a larger key proves it complete) and flushes the last
  key at end of input. This is the causal emit-on-boundary + flush pattern already
  built for `resample`; it introduces a one-key-group lag in live emission
  (batch is unaffected) and is the honest price of "settled state per timestamp."

So `combine_latest` cardinality is a single predictable quantity - distinct keys
in the union - identical across batch, streaming, and graph, differing only in
*when* the live path emits.

### Implementation note

Coalescing is a "collapse consecutive same-key rows, keep the last" step over the
existing C++ aligner output: a post-pass on the collected array for batch
(eager + `run_batch`), a generator that emits on key-advance + at end for
`combine_latest_iter`, and - because a graph `combine_latest` feeds downstream
graph nodes per event - an emit-on-key-advance + `flush` behavior on the
`dag::CombineLatestNode` (reusing the resample flush infrastructure). This is a
targeted change to the combine node, not a broad engine rewrite; the resampler,
functors, and DAG compiler are untouched.

## `merge` / `split`

`merge` is **not** an alignment op and does **not** coalesce - it faithfully
interleaves *every* event in key order, tagging each with its source (both rows
at a shared key are kept). Raw-array oriented (source tags are integral):
`merge(*values, index=None) -> (values, sources, index)` (index None when
positional; positional/indexed mixing raises as in combine_latest).
`split(values, sources, index=None, n=None) -> list[(values, index)]` is its
inverse.

## Other combinators

Single-stream shape ops - no alignment, no coalescing, no mixing concern. Apply
the polymorphic wrapper (Node -> graph; else regime -> one `Stream` -> existing
logic on `.values`, threading `.index` through the same mask/projection -> adapt):

- `dropna(values, index=None, how="any")` -> `(values, index)` / `Stream` / `Node`.
- `select(values, columns, index=None)` -> `(values, index)` / `Stream` / `Node`.
- `filter(values, predicate, index=None)` -> `(values, index)` / `Stream` (eager
  only; `Node` raises the existing "not supported" message).
- `resample(values, index=None, *, every=None, count=None, agg="last", origin=0,
  label="left")` -> `(values, index)` / `Stream` / `Node`, where the returned
  `index` is the bar labels. **`width` is renamed `every`** (reads as "resample
  every 60"; matches polars). Engine keeps its internal `width` argument.

## Streaming twins and `pace`

`*_iter` forms yield **`(value, index)`** per event (values first; `index` None
when positional), the scalar analog of the batch `(values, index)` return.
`combine_latest_iter` additionally coalesces (emit on key-advance + at end).
`pace(*values, index=None, speed=1.0, sleep=None)` yields `(value, index, source)`
events. A keyless live feed yields `(value, None)` (or arrival-stamped) events.

## `Stream` type

```python
class Stream:
    def __init__(self, values, index=None): ...   # values (T,) or (T, N); index (T,) or None
    values          # np.ndarray
    index           # np.ndarray or None  (None == positional)
    def __len__(self): ...
    @classmethod
    def from_pandas(cls, series_or_frame): ...     # data -> values, pandas index -> index
    def to_pandas(self): ...                        # Series/DataFrame; positional -> RangeIndex
```

Thin wrapper over two arrays (`__slots__`), zero per-element overhead, never in
the hot loop. Minimal surface for v1: `values`, `index`, `len`, pandas interop.

## DAG boundary + rename

`_as_stream` / `Dag(...)` accept bare arrays, `Stream`s, or `(values, index)`;
positional maps to the engine's row-number keys at push. `Dag` outputs follow
`align_outputs` in the new `(values, index)` shape. `_align_results` uses the new
`combine_latest(*values, index=[...])` signature. Sweep `key(s)` -> `index` across
the public Python surface and docs; the C++ `Key` template and internal `key`
variables stay.

## Testing

- **Identity preserved and strengthened:** batch == stream == graph == eager
  oracle for `combine_latest` under coalescing (positional lockstep, indexed
  distinct-key, tie-coalescing, `emit` variants), across every combinator.
- **Cardinality:** assert "# distinct keys" for indexed, N for equal-length
  positional, an error for unequal-length no-index, and per-arrival for a keyless
  live/`_iter` run.
- **Type mirroring:** raw -> `(values, index)` (index None positional), `Stream`
  -> `Stream`, `Node` -> `Node`, for every combinator.
- **Coalescing:** the phantom same-key row is gone (indexed `[1,2,3,4,4]` ->
  `[1,2,3,4]`); the trailing key flushes in streaming.
- **Interop:** `Stream.from_pandas`/`to_pandas` round-trip.

## Non-goals

- No changes to the resampler, functors, or DAG compiler (only the combine node
  gains coalesce + flush).
- No new combinators; overlapping/session windows, quantile/vwap, upsample remain
  future work.

## Decisions log (locked)

1. Stream = values + optional index; values-primary; `key` -> `index`.
2. Polymorphic in/out: raw -> `(values, index)`; `Stream` -> `Stream`; `Node` ->
   `Node`. Raw return always a 2-tuple; `index=None` positional (no allocation).
3. `combine_latest` emits one row per distinct key (coalesce same-key). Single
   behavior, no per-event flag.
4. No index = aligned clocks = lockstep; **equal length required** (unequal is an
   error). Different clocks require an index.
5. Live keyless async -> arrival-order key stamped on receipt -> emit per arrival;
   capture keys for reproducible backtests.
6. `merge`/`split` never coalesce; raw-array oriented.
7. `resample` `width` -> `every`.
8. `_iter` events are `(value, index)`; `pace` events `(value, index, source)`.
9. `Stream` v1 surface: `values`, `index`, `len`, `from_pandas`/`to_pandas`.
10. `key` retired everywhere, incl. the internal C++ `Key` template -> `Index`.
11. "combinator" retired for "stream operator", in prose and code identifiers.
12. "event" (streaming) and "row" (batch) kept and used precisely; "stream" is
    the single concept word ("series"/"feed" retired). See the Vocabulary table.
