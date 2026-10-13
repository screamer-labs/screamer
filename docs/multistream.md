# Streams, values, and alignment

screamer's single-stream functors (`RollingMean`, `RollingCorr`, ...) assume
lockstep alignment: row `i` of one input pairs with row `i` of another. Real
multi-stream data breaks that assumption - feeds tick at different rates, arrive
out of step, and drop samples. screamer adds a small, composable set of
operators for combining, splitting, and filtering streams that
do **not** tick together, while keeping every existing functor unchanged.

The whole design rests on five principles.

## 1. A stream is values with an optional index

A **stream** is a sequence of **values** (a 1-D or 2-D NumPy array), optionally
annotated with an **index** (a 1-D array of the same length). The index is an
ordering coordinate - a `datetime64` timestamp, an `int64` tick count, a
`float64` second - that locates each value in a shared timeline. It is not a
dict entry; screamer only ever *orders* and *compares* index values.

**`index=None` means positional.** A positional stream has no explicit ordering
coordinate; its position (row number) is its logical place in time. Two
positional streams are assumed to tick together, so their rows pair by position
and must have equal length. To model streams on *different* clocks, provide an
index for each stream.

A stream value is either a bare ndarray (positional) or a `(values, index)` 2-tuple
(indexed). `from_pandas` / `to_pandas` convert between these and pandas objects:

```python
from screamer import to_pandas, from_pandas

values, index = from_pandas(df)       # DataFrame -> (values, index) tuple
series = to_pandas(values, index)     # (values, index) -> Series or DataFrame
```

See the [tuple convention guide](functions_streams/Stream.md) for the full contract
and the pandas round-trip.

## 2. Stream operators are polymorphic

Most stream operators (`CombineLatest`, `Dropna`, `Filter`, `Select`,
`Resample`) dispatch on the type of their inputs and mirror that type on return:

| Input type | Return type |
|---|---|
| bare ndarray(s), optional `index=` | `(values, index)` 2-tuple; `index is None` when positional |
| `(values, index)` tuple(s) | `(values, index)` 2-tuple |
| graph `Node`(s) | a `Node` (builds the Pipeline) |

Batch return is always a `(values, index)` 2-tuple: `vals, idx = CombineLatest()(a, b)` always
works; `idx is None` is a checkable flag meaning "no real ordering here." A bare
array is treated as a positional stream with `index=None`.

`Merge` and `split` follow the same input convention but their *outputs*
differ because they carry a per-event `sources` tag. `Merge` returns
`(values, sources, index)`, where `sources` is a `uint32` array whose `i`-th
entry is the index of the input stream that event `i` came from. `split` inverts
this, returning per-source `(values, index)` pairs.

## 3. Compute functors preserve cardinality; stream operators may change it

| Layer | Cardinality | Examples |
|---|---|---|
| **Compute functors** | preserved (output length == input length) | `RollingMean`, `RollingCorr`, `FillNa`, `Ffill` |
| **Stream operators** | may change it | `Merge`, `CombineLatest`, `Dropna`, `Filter`, `split` |

Compute functors handle `NaN` internally via their `nan_policy` (see
[NaN and warmup](nan_and_warmup.md)) and never add or drop rows. Stream operators own
all time alignment and stream shaping. `Dropna` / `Filter` / `split` are the
cardinality-changing tools; `fillna` / `ffill` are shape-preserving and belong
to both worlds.

## 4. Alignment is a separate layer from computation

Time-aware stream operators do the index handling and hand *aligned* data to the
unchanged compute functors. The idiom is:

```python
from screamer import CombineLatest, RollingCorr

# Two async price streams, each a (values, index) pair.
aligned, idx = CombineLatest()(p_a, p_b, index=[t_a, t_b])   # as-of latest-value join
corr = RollingCorr(20)(aligned[:, 0], aligned[:, 1])         # functor, untouched
```

`CombineLatest` emits **one row per distinct index** (same-index events from
different streams coalesce into a single settled row). `emit="when_all"`
(default) waits until every input is warm; `emit="on_any"` emits from the first
event with `NaN` for inputs not yet seen. Feed the aligned columns to any
existing functor.

### Multi-column streams

`CombineLatest` concatenates the columns of its inputs, which are the fields of
each event. A `(T, 4)` bar stream and a `(T,)` signal give `(T, 5)` rows of
`[open, high, low, close, signal]`, and inside a `Pipeline` an OHLC node joins
a signal node the same way:

```python
rows, idx = CombineLatest()(bars, signal)     # (T, 5)
```

By default the columns of a 2-D stream are fields. When they are assets
instead, for example a `(T, K)` panel of prices next to one `(T,)` market
series, say so with `layout`, one layout string per input. The asset axes
broadcast, so the market series is paired with every asset, and the fields
concatenate:

```python
rows, idx = CombineLatest(layout=("TA", "T"))(panel, market)   # (T, K, 2)
```

A layout applies to arrays, `(values, index)` tuples, and iterators of
events. Inside a `Pipeline`, streams carry fields only and any layout raises
`ValueError`.

In JavaScript, `combineLatest` takes the inputs as one array and the options
as an object, with the same join. An indexed input is an object: `{values,
index}` for an array, and `{value, index}` for an event of an iterator, as
the tuple marks it in Python. A plain Array is always values. Arrays and
`{values, index}` inputs give `{values, index}`, with the values an
`NdArray`; iterators and async iterables give an iterator of `{value, index}`
rows:

```js
const { values } = combineLatest([panel, market], { layout: ["TA", "T"] });   // values.shape [T, K, 2]
const joined = combineLatest([{ values: a, index: ia }, { values: b, index: ib }]);
```

Inside a `Pipeline`, `emit="on_any"` can emit a row before a multi-column input
has delivered its first event, for example when a `Dropna` upstream drops that
input's first rows. The row's width for that input is then unknown, and the
`Pipeline` raises `ValueError` once the input arrives. Use the default
`emit="when_all"` for such inputs.

### Resampling each column

`Resample` with a built-in aggregation (`first`, `last`, `min`, `max`, `sum`,
`count`, `mean`, `ohlc`) aggregates every column of a multi-column stream
separately. An aggregation with several outputs appends them as a last axis:
`agg="ohlc"` turns `(T, X)` values into `(T', X, 4)` bars, one set of bars per
column. Inside a `Pipeline` the result is flat, `(T', 4 * X)`, with the four
bar fields of each column side by side (`[o0, h0, l0, c0, o1, ...]`), so
`values.reshape(len(values), X, 4)` gives the eager result. A functor `agg`
aggregates one column, and multi-column input raises `ValueError`.

```python
bars, idx = Resample(freq=60_000, agg="ohlc")(panel, t)   # (T', K, 4)
```

Other stream operators:

- `Merge()(*values, index=None)` -> `(values, sources, index)`: one index-sorted,
  source-tagged stream. Wide streams merge row by row; every input has the
  same row width.
- `split(values, sources, index=None)` -> the inverse of `Merge`.
- `Dropna(how="any")(values, index=None)` -> drop NaN events.
- `Filter()(data, mask)` -> keep each data value whose aligned mask is nonzero.
Every stream operator is polymorphic over its input: pass lazy iterators of
events and it returns a lazy iterator; pass arrays or `(values, index)` tuples
and it returns the batch result. No separate `*_iter` function is needed.
(`split` has no streaming form.)

## 5. Causality

- **Causal**: an output at index `t` depends only on events at indices `<= t`. There
  is no backward-fill and no lookahead operator, ever. This guarantee holds whether
  you run the operator on a stored dataset or feed it events one at a time live.
  Conformance is enforced by `tests/test_streams_identity.py`.

## See also

- [Polymorphic API](polymorphic_api.md) - the single-stream input/output
  contract; lockstep is the positional (no-index) special case of this page.
- [Pipelines](pipelines.md) - wiring stream operators and functions into
  a graph you define once and run batch or live.
- [NaN and warmup](nan_and_warmup.md) - how compute functors treat `NaN`.
