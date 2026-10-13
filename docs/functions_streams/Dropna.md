---
name: Dropna
title: Dropna
kind: class
short: Drop events whose value is NaN.
tags:
- dropna
- drop missing
- nan
see_also:
- Filter
- FillNa
- Ffill
topics:
- missing-data
- streams
covers:
- dropna
---

# `Dropna`

Drop events whose value is `NaN`. This changes the length of the stream (it is a
cardinality-changing stream operator, unlike the shape-preserving compute
functors). Usable eagerly and inside a `Pipeline`.

Feeding a lazy iterator of `(value, index)` pairs returns a lazy iterator of the surviving events; feeding arrays or `(values, index)` tuples returns the batch result.

Each argument is one stream; an index goes in a `(values, index)` tuple (or
`index=` for one array). Several streams merge in index order, ties in argument
order, and each event is dropped on its own, as the `Dropna` node of a
`Pipeline` given several streams does: `Dropna()(a, b)` with `a` and `b` arrays
or tuples is that node.

<!-- HELP_END -->

## Examples

The event at index 2 is `NaN`, so it is removed; the surviving values are
returned values-first.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Dropna

   vals = np.array([1.0, np.nan, 3.0])
   idx  = np.array([1, 2, 3])

   clean_vals, clean_idx = Dropna()(vals, index=idx)
   print(clean_vals)
   print(clean_idx)
```

Two streams: their events merge in index order, and each `NaN` event is dropped.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Dropna

   a = (np.array([1.0, np.nan, 3.0]), np.array([0, 2, 4]))
   b = (np.array([np.nan, 2.0, 4.0]), np.array([1, 2, 3]))

   vals, idx = Dropna()(a, b)
   print(vals)
   print(idx)
```
