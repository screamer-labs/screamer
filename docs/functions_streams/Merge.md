---
name: Merge
title: Merge
kind: class
short: Merge N value streams into one index-sorted (values, sources, index).
tags:
- merge
- interleave
- union
- multiplex
see_also:
- split
- CombineLatest
- Select
topics:
- streams
covers:
- merge
---

# `Merge`

Interleave several streams into one index-sorted stream, tagging each event with
the integer index of the source it came from. This is the causal, order-preserving
way to fan several streams into a single timeline.

<!-- HELP_END -->

Feeding `Merge` lazy iterators of `(value, index, source)` events returns a lazy
iterator (each source is numbered by a per-source arrival counter); feeding
arrays or `(values, index)` tuples returns the eager `(values, sources, index)` 3-tuple.

## Examples

Two streams are merged into one stream ordered by index. `sources` records which
stream each event came from.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Merge

   a_v = np.array([1.0, 3.0])
   a_k = np.array([1, 3])
   b_v = np.array([2.0, 4.0])
   b_k = np.array([2, 4])

   values, sources, idx = Merge()(a_v, b_v, index=[a_k, b_k])
   print(list(zip(idx.tolist(), sources.tolist())))
```

`split` is the inverse: it restores the original per-source streams.

## Wide streams

A stream of rows, such as a `(T, K)` array, merges row by row, in the same
order as a 1-D stream: by index, and at equal indexes in input order (a lower
input first). The merged values have shape `(rows, K)`, or `(rows, *A)` for
rows of any shape `A`. Every input must have the same row shape: a `(T, 2)`
stream and a `(T, 3)` stream, or a 1-D stream and a `(T, 2)` stream, raise
`ValueError` naming both widths. An empty input of the same row shape, such as
`np.empty((0, K))`, adds no rows. Lazy iterators of row events give one
`(row, index, source)` event per row, the row an array of the event's shape.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Merge

   a = np.array([[1.0, 10.0], [3.0, 30.0]])
   b = np.array([[2.0, 20.0]])
   values, sources, idx = Merge()((a, [1, 3]), (b, [3]))
   print(values.tolist(), sources.tolist(), idx.tolist())
```

Each `(values, index)` input needs one index value per row; an index of
another length raises `ValueError` naming both lengths.

## JavaScript

`merge(inputs)` takes an array of inputs: arrays (a typed array, an Array, T
rows or an `NdArray`) or `{ values, index }` streams, and returns
`{ values, sources, index }` (a `Float64Array`, or an `NdArray` for wide
inputs; a `Uint32Array`; and the merged index, or `null` without one).
Iterators and async iterables of events (a value, or `{ value, index }`)
return an iterator of `{ value, index, source }` events. The order, the width
rule and the errors are the same as in Python, with `RangeError` for
`ValueError`.
