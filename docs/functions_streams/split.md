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

Event `i` goes to output `sources[i]`, in event order. A source is a whole
number from 0 to 4294967295; any other source raises `TypeError`. `n` sets the
number of outputs, by default `max(sources) + 1`; an `n` at or below the largest
source raises `ValueError`, since events would be dropped.

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
as an option, and returns `[values, index]` pairs of `Float64Array`s (`index`
is `null` without one), with the same values as Python. An `n` that would drop
an event raises `RangeError`.
