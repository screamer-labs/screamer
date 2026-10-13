---
name: split
title: split
kind: function
short: Partition a merged tagged stream back into per-source streams.
tags:
- split
- demultiplex
- partition
see_also:
- Merge
- Select
- CombineLatest
topics:
- streams
---

# `split`

The inverse of `Merge`: partition a source-tagged stream back into one stream per
source. `split(*Merge()(a_v, b_v, index=[a_k, b_k]))` reconstructs the original
streams.

Event `i` goes to output `sources[i]`, in event order. `values` may have any
rank with time first, `(T, ...)`: each event is a row, and each output keeps the
row shape, `(rows, ...)`. A source is a whole number: a bool, a fractional or a
non-finite source raises `TypeError`. It is from 0 to 2147483646, one less than
the most outputs (2**31 - 1); another source raises `ValueError`. `n` sets the
number of outputs, by default `max(sources) + 1`; an `n` at or below the largest
source raises `ValueError`, since events would be dropped. A split makes at most
2\*\*20 (1048576) outputs: more, from a large source or `n`, raises `ValueError`
before anything is allocated.

<!-- HELP_END -->

```{eval-rst}
.. autofunction:: screamer.split
```

## Examples

Merge two streams, then split them apart again and read back the first source.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Merge, split

   a_v = np.array([1.0, 3.0])
   a_k = np.array([1, 3])
   b_v = np.array([2.0, 4.0])
   b_k = np.array([2, 4])

   merged_v, merged_s, merged_k = Merge()(a_v, b_v, index=[a_k, b_k])
   streams = split(merged_v, merged_s, index=merged_k)
   print(streams[0][0])   # values of source 0
```

## JavaScript

`split(values, sources, { index, n })` takes the same arguments, with the index
as an option, and returns `[values, index]` pairs with the same values as
Python. `values` is a `Float64Array` for 1-D values, or an `NdArray` of shape
`[rows, ...]` for values given as an `NdArray` of rank 2 or more or as rows;
`index` is a `Float64Array`, or `null` without one. Where Python raises
`ValueError`, JavaScript raises `RangeError`.
