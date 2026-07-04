# Stream / index model redesign

**Status:** design under review (pre-release; no back-compat required)
**Scope:** Fix the multi-stream *interface* so it reflects the real model: a
stream is a sequence of **values**; an **index** (ordering coordinate) is
optional metadata, not a required key and not a dict. Introduce a `Stream` type,
make the combinators polymorphic on input type (raw arrays / `Stream` / graph
`Node`) with a mirrored return type, rename `key(s)` -> `index` everywhere
user-facing, and default a missing index to position with **no allocation**.
The engine and all causal/identity guarantees are unchanged; this is an
interface and vocabulary redesign.

## Motivation

The current docs and API define a stream as "a sequence of `(key, value)`
events" and every combinator takes and returns `(keys, values)` pairs with the
key mandatory and first. That framing is wrong in three ways:

- **A stream is values.** `RollingMean(50)(prices)` needs no key. The coordinate
  is *when/where* each value sits, not part of the payload.
- **"key" imports dict semantics.** There is no lookup; the coordinate can
  repeat; the only operations are order and compare. It is an ordering axis, not
  an associative key.
- **The coordinate is optional.** It is needed only to align streams that tick on
  different clocks (`combine_latest`, `merge`, time-`resample`), and even then it
  can be implicit (position) or any counter. Today the combinators *require* it
  (`combine_latest(a, b)` on bare arrays is rejected), while the DAG layer already
  defaults it to row number - an inconsistency.

The engine underneath is correct (ordered arrays, coordinate never interpreted,
row-number default). Only the interface is wrong. Since the multi-stream layer is
unreleased, we fix it now rather than ship a shape we would have to break later.

## The model

- A **stream** is `values` (a 1-D or 2-D numpy array), optionally annotated with
  an **index** (a 1-D array of the same length): timestamps, a tick/message
  counter, or any orderable coordinate. **`index=None` means position** (row
  number / arrival order) and allocates nothing.
- **Values are primary; the index is optional metadata.** Every signature leads
  with values; the index is a keyword.
- The coordinate is called **`index`** everywhere user-facing (array name,
  keyword, `Stream.index`). The engine's internal `int64` order key is unchanged
  and stays internal.

## Polymorphic in / out contract

Combinators dispatch on input type and mirror it on the way out - the same
contract the single-series functors already use (scalar/array/iterator/`Node`):

| You pass | You get back |
|---|---|
| raw value array(s), optional `index=` | `(values, index)` - a 2-tuple; `index is None` when positional |
| `Stream` object(s) | a `Stream` |
| graph `Node`(s) | a `Node` (builds the DAG, unchanged) |

Rules:
- **One core.** Inputs normalize to `Stream` internally; the existing core runs
  once; the result is adapted back to match the input regime. A single shared
  dispatch helper does this for every combinator (no per-combinator duplication).
- **Raw return is always `(values, index)`** (values first, the one you usually
  want), 2-tuple shape regardless of data, with `index=None` for the positional
  case so nothing is allocated. `vals, idx = combine_latest(a, b)` always works;
  `idx is None` is a checkable "no real ordering here."
- **Mixed inputs:** a bare array auto-wraps to a position-indexed `Stream`, so
  mixing is legal; the output is a `Stream` if *any* input was a `Stream`, else
  raw. `Node` in any position routes to the graph builder (as today).

## The `Stream` type (new)

```python
class Stream:
    def __init__(self, values, index=None): ...   # values: (T,) or (T, N); index: (T,) or None
    values          # np.ndarray
    index           # np.ndarray or None  (None == positional)
    def __len__(self): ...
```

Proposed interop surface (confirm in review):
- `Stream.from_pandas(series_or_frame)` -> values from the data, `index` from the
  pandas index.
- `Stream.to_pandas()` -> `pd.Series`/`DataFrame` (positional index when
  `index is None`).
- Iteration (`for value, idx in stream` or over events) - only if we want a
  per-event view; otherwise omit.

`Stream` is a thin wrapper over two arrays: zero per-element overhead, and it
never enters the hot streaming loop (the engine passes `value`/`key` scalars).

## Per-combinator signatures (before -> after)

Raw form shown; each also accepts `Stream`(s) (-> `Stream`) and `Node`(s) (-> `Node`).

- **combine_latest**
  - before: `combine_latest(*series, emit="when_all", func=None)` -> `(keys, aligned)`
  - after: `combine_latest(*values, index=None, emit="when_all", func=None)`
    -> `(aligned_values, index)`. `index` is a list of per-stream index arrays
    (or `None` for all-positional); the returned `index` is the merged clock
    (`None` if all inputs were positional).
- **dropna**
  - before: `dropna(keys, values=None, how="any")`
  - after: `dropna(values, index=None, how="any")` -> `(values, index)`.
- **select**
  - before: `select(keys, values=None, columns=None)`
  - after: `select(values, columns, index=None)` -> `(values, index)`.
- **filter** (eager only; no graph form)
  - before: `filter(keys, values=None, predicate=None)`
  - after: `filter(values, predicate, index=None)` -> `(values, index)`.
- **resample**
  - before: `resample(keys, values=None, *, width=None, count=None, agg=..., origin=0, label="left")`
  - after: `resample(values, index=None, *, every=None, count=None, agg="last", origin=0, label="left")`
    -> `(values, index)` where the returned `index` is the bar labels.
    **Proposed rename `width` -> `every`** (reads as "resample every 60"; matches
    polars). Confirm in review.
- **merge / split** (the structural pair - see open decision)
  - before: `merge(*series) -> (keys, values, sources)`; `split(keys, values, sources, n=None)`
  - after (proposed): `merge(*values, index=None) -> (values, sources, index)`;
    `split(values, sources, index=None, n=None) -> list[(values, index)]`. These
    stay raw-array oriented because the source tags are integral array metadata;
    a `Stream` form would return `(Stream, sources)`.
- **pace** (async replay)
  - before: `pace(*series, speed=1.0, sleep=None)` -> yields `(key, value, source)`
  - after: `pace(*values, index=None, speed=1.0, sleep=None)` -> yields events
    (event shape open - see decision).

### `_iter` streaming twins

`merge_iter`, `combine_latest_iter`, `dropna_iter`, `filter_iter`, `select_iter`,
`resample_iter` are the per-event forms. Proposed: each event is `(value, index)`
(values first, `index` None when positional), consuming/producing the same shape.
A live feed with no timestamps yields bare values (index None). Confirm the event
shape in review.

## DAG boundary

`_as_stream` and `Dag(...)` already accept bare value arrays (row-number index);
they adopt the same normalization (bare array / `Stream` / `(values, index)` ->
internal `Stream`; `index=None` -> the engine's row-number keys at push time).
The graph path (`Node` in -> `Node` out) is unchanged. `Dag` feeds may be bare
arrays or `Stream`s; outputs follow `align_outputs` as today, in the new
`(values, index)` shape.

## Naming migration

`key(s)` -> `index` across the public surface: combinator params, return values,
docstrings, `Stream.index`, and all docs (the `functions_streams/*` pages,
`multistream.md`, notebooks 07-10). The engine's C++ `Key` template and internal
`key` variables stay (internal, not user-facing). `merge`/`split` keep `sources`.

## Testing

- **Identity preserved:** the existing batch == streaming == oracle guarantees
  must still hold; the redesign is interface-only, so the core results are
  unchanged. Update the existing tests to the new signatures.
- **Type mirroring:** for each combinator, assert raw in -> `(values, index)`
  out (with `index is None` for positional and an array when supplied), `Stream`
  in -> `Stream` out, `Node` in -> `Node` out.
- **No allocation for positional:** assert the positional path returns
  `index is None` (not a materialized `arange`).
- **Interop:** `Stream.from_pandas`/`to_pandas` round-trips (if included).

## Open decisions to confirm in review

1. **`resample` `width` -> `every`** rename (recommended) or keep `width`.
2. **merge/split** stay raw-array-only (recommended) or also get a `Stream` form.
3. **`_iter` / `pace` event shape**: `(value, index)` per event (recommended) vs
   bare `value` vs `(index, value)`.
4. **`Stream` interop surface**: include `from_pandas`/`to_pandas` and/or
   iteration now, or keep `Stream` minimal (`values`/`index`/`len`) for v1.

## Non-goals

- No engine changes (the compiled DAG, causal emit, flush, reducers are
  untouched).
- No new combinators; this is purely the interface/vocabulary of the existing
  ones.
