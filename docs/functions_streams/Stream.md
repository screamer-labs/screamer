---
name: Stream
title: Stream
kind: function
short: A sequence of values with an optional ordering index.
topics:
- streams
---

# `Stream`

The object form of a stream: a values array paired with an optional index (the
ordering coordinate). Instead of passing and receiving loose `(values, index)`
tuples, you can pass a `Stream` and get a `Stream` back, with the index carried
along. Accepted by every stream operator that preserves the stream shape.

<!-- HELP_END -->

```python
from screamer import Stream
```

`Stream` is also importable from `screamer.streams`, where it is defined; the
top-level `screamer` re-exports it. Prefer `from screamer import Stream`.

```{eval-rst}
.. autoclass:: screamer.streams.Stream
   :members:
```

## Constructor

`Stream(values, index=None)`

- **`values`** — a 1-D array of shape `(T,)`, or a 2-D array of shape `(T, N)`.
  For a 2-D array the `N` columns are `N` parallel series.
- **`index`** — a 1-D array of length `T`, or `None`. `None` means
  **positional**: the row number is the ordering coordinate and nothing is
  stored. The index is an *ordering coordinate* — a `datetime64` timestamp, an
  `int64` tick counter, a `float64` second — used only to order and compare
  events. It is not a lookup key and has no dict semantics.
- Raises `ValueError` if `index` is given and its length does not match
  `values`.

## Attributes

- **`.values`** — the values array.
- **`.index`** — the index array, or `None` when the stream is positional.

## Methods

- **`len(stream)`** — the number of rows, `T`.
- **`Stream.from_pandas(series_or_dataframe)`** — build a `Stream` from pandas.
  The pandas data becomes `.values` and the pandas index becomes `.index`. A
  plain `RangeIndex` is kept as a numbered index, **not** converted to `None`;
  pass a positional `Stream` explicitly if you want `index=None`.
- **`stream.to_pandas()`** — return a `Series` (1-D values) or `DataFrame` (2-D
  values). A positional stream is given pandas' default index.

## How it fits the operator API

Stream operators accept three input forms and mirror the form on return:

| You pass | You get back |
|---|---|
| raw array(s), with optional `index=` | `(values, index)` tuple (`index is None` when positional) |
| a `Stream` | a `Stream` |
| a graph `Node` | a `Node` (builds the DAG) |

So `combine_latest`, `dropna`, `select`, `Filter`, and `resample` take a
`Stream` and return a `Stream`. A bare array is treated as positional
(`index=None`). `merge` and `split` are the exception: they work on raw tagged
arrays, not on `Stream` objects.

## Example

Build an indexed `Stream` and pass it straight to an operator, which returns a
`Stream`.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Stream, resample
   # --- hide: stop ---
   s = Stream(np.array([1.0, 2.0, 3.0, 4.0]), index=np.array([0, 3, 10, 12]))

   bars = resample(s, every=10, agg="mean")   # a Stream in, a Stream out
   print(bars.values, bars.index)
```

## See also

- [Streams, values, and alignment](../multistream.md) — the conceptual model
  behind streams, indices, and alignment.
