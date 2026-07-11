---
name: dropna
title: dropna
kind: function
short: Drop events whose value is NaN.
topics:
- missing-data
- streams
covers:
- Dropna
---

# `dropna`

Drop events whose value is `NaN`. This changes the length of the stream (it is a
cardinality-changing stream operator, unlike the shape-preserving compute
functors). Usable eagerly and inside a `Dag`.

Feeding a lazy iterator of `(value, index)` pairs returns a lazy iterator of the surviving events; feeding arrays or a `Stream` returns the batch result.

<!-- HELP_END -->

```{eval-rst}
.. autofunction:: screamer.streams.dropna
```

## Example

The event at index 2 is `NaN`, so it is removed; the surviving values are
returned values-first.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import dropna
   # --- hide: stop ---
   vals = np.array([1.0, np.nan, 3.0])
   idx  = np.array([1, 2, 3])

   clean_vals, clean_idx = dropna(vals, index=idx)
   print(clean_vals)
   print(clean_idx)
```
