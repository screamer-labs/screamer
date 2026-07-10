# Streams, values, and alignment

screamer's single-stream functors (`RollingMean`, `RollingCorr`, ...) assume
lockstep alignment: row `i` of one input pairs with row `i` of another. Real
multi-stream data breaks that assumption - feeds tick at different rates, arrive
out of step, and drop samples. The `screamer.streams` module adds a small,
composable layer for combining, splitting, filtering, and replaying streams that
do **not** tick together, while keeping every existing functor unchanged.

The whole design rests on four principles.

## 1. A stream is values with an optional index

A **stream** is a sequence of **values** (a 1-D or 2-D NumPy array), optionally
annotated with an **index** (a 1-D array of the same length). The index is an
ordering coordinate - a `datetime64` timestamp, an `int64` tick count, a
`float64` second - that locates each value in a shared timeline. It is not a
a dict entry and has no dict semantics; screamer only ever *orders* and *compares*
index values.

**`index=None` means positional.** A positional stream has no explicit ordering
coordinate; its position (row number) is its logical place in time. Two
positional streams are assumed to tick together - they are aligned clocks - so
their rows pair by position. To model streams on *different* clocks, provide an
index for each stream; without one, aligned clocks are assumed and the lengths
must match.

The `Stream` type is a thin wrapper over two arrays (`.values` and `.index`)
with no per-element overhead, plus `from_pandas` / `to_pandas` converters:

```python
from screamer import Stream

s = Stream(values, index=None)         # positional
s = Stream(values, index=timestamps)   # indexed
```

See the [`Stream` reference](functions_streams/Stream.md) for the full contract:
constructor, shape rules, attributes, methods, and the pandas round-trip.

## 2. Stream operators are polymorphic

Most stream operators (`combine_latest`, `dropna`, `filter`, `select`,
`resample`) dispatch on the type of their inputs and mirror that type on return:

| Input type | Return type |
|---|---|
| raw value array(s), optional `index=` | `(values, index)` 2-tuple; `index is None` when positional |
| `Stream` object(s) | a `Stream` |
| graph `Node`(s) | a `Node` (builds the DAG) |

Raw return is always values-first: `vals, idx = combine_latest(a, b)` always
works; `idx is None` is a checkable flag meaning "no real ordering here." A bare
array auto-wraps to a positional `Stream`; if any input was a `Stream` the
output is also a `Stream`.

`merge`, `split`, and `replay` accept `Stream` inputs too, but their *outputs*
differ because they carry a per-event `sources` tag: `merge` returns
`(values, sources, index)` and `replay` yields `(value, index, source)` events.
`split` given a `Stream` returns a list of `Stream`s (one per source); given raw
arrays it returns `(values, index)` pairs.

## 3. Compute functors preserve cardinality; stream operators may change it

| Layer | Cardinality | Examples |
|---|---|---|
| **Compute functors** | preserved (output length == input length) | `RollingMean`, `RollingCorr`, `FillNa`, `Ffill` |
| **Stream operators** | may change it | `merge`, `combine_latest`, `dropna`, `filter`, `split`, `replay` |

Compute functors handle `NaN` internally via their `nan_policy` (see
[NaN and warmup](nan_and_warmup.md)) and never add or drop rows. Stream operators own
all time alignment and stream shaping. `dropna` / `filter` / `split` are the
cardinality-changing tools; `fillna` / `ffill` are shape-preserving and belong
to both worlds.

## 4. Alignment is a separate layer from computation

Time-aware stream operators do the index handling and hand *aligned* data to the
unchanged compute functors. The idiom is:

```python
from screamer import combine_latest, RollingCorr

# Two async price streams, each a (values, index) pair.
aligned, idx = combine_latest(p_a, p_b, index=[t_a, t_b])   # as-of latest-value join
corr = RollingCorr(20)(aligned[:, 0], aligned[:, 1])         # functor, untouched
```

`combine_latest` emits **one row per distinct index** (same-index events from
different streams coalesce into a single settled row). `emit="when_all"`
(default) waits until every input is warm; `emit="on_any"` emits from the first
event with `NaN` for inputs not yet seen. Feed the aligned columns to any
existing functor.

Other stream operators:

- `merge(*values, index=None)` -> `(values, sources, index)`: one index-sorted,
  source-tagged stream.
- `split(values, sources, index=None)` -> the inverse of `merge`.
- `dropna(values, index=None, how="any")` / `filter(values, predicate, index=None)`
  -> drop events.
- `replay(*values, index=None, speed=1.0)` -> async replay; `speed=inf` is a
  max-speed backtest. Yields `(value, index, source)` per event.

`merge_iter` and `combine_latest_iter` yield `(value, index)` events one at a
time. (`split` has no streaming form, and `replay` is itself the
streaming/replay driver.)

`resample`, `dropna`, `filter`, and `select` are unified: pass a lazy iterator
of `(value, index)` pairs and the operator returns a lazy iterator; pass arrays
or a `Stream` and the operator returns the batch result. No separate `*_iter`
function is needed.

## Migration from the retired `*_iter` names

The per-operator streaming variants were removed in v0.5. The unified operators
handle both batch and lazy inputs:

| Old (removed) | New |
|---|---|
| `resample_iter(events, ...)` | `resample(events, ...)` |
| `dropna_iter(events)` | `dropna(events)` |
| `filter_iter(events, pred)` | `filter(events, pred)` |
| `select_iter(events, cols)` | `select(events, cols)` |

## 5. Causal, and identical across modes

- **Causal**: an output at index `t` depends only on events at indices `<= t`. There
  is no backward-fill and no lookahead operator, ever.
- **Batch == streaming == graph**: the batch form and its streaming twin emit
  byte-identical event sequences; `replay` changes only *when* events are emitted,
  never their values or order. This is what lets you validate a pipeline on
  stored data and run the identical pipeline live. It is enforced by the
  identity matrix in `tests/test_streams_identity.py`.

## See also

- [Polymorphic API](polymorphic_api.md) - the single-stream input/output
  contract; lockstep is the positional (no-index) special case of this page.
- [The computational graph](dag.md) - wiring stream operators and functions into
  a graph you define once and run batch or live.
- [NaN and warmup](nan_and_warmup.md) - how compute functors treat `NaN`; `ffill`
  is the same forward-fill carry that `combine_latest` uses.
