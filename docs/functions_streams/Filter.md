---
name: Filter
title: Filter
kind: class
short: 2-input mask gate - keep each data value whose aligned mask is nonzero.
topics:
- streams
---

# `Filter`

A 2-input mask gate: keep each data value whose aligned mask value is nonzero
(zero or NaN drops, any other value keeps). The mask is an ordinary stream
built from upstream comparison or logic ops. No Python predicate or callback -
the gate logic runs entirely in C++ for batch, lazy, and graph regimes.

Gate rule: `mask == 0` or `isnan(mask)` drops the aligned data value; any
other mask value (positive, negative, non-NaN) keeps it. A NaN data value
passes through unmodified when its aligned mask is nonzero.

<!-- HELP_END -->

## Example

Keep the positive values by building a mask with `GreaterThan`.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Filter, GreaterThan

   x = np.array([5.0, -2.0, 8.0, -1.0, 3.0])
   mask = GreaterThan()(x, np.zeros_like(x))
   survivors, idx = Filter()(x, mask)
   print(survivors)
```

## Three regimes, byte-identical results

Batch (arrays or Streams)::

    survivors, idx = Filter()(data_array, mask_array)

Lazy (iterators of (value, index) pairs)::

    for val, idx in Filter()(iter(data_events), iter(mask_events)):
        ...

Graph (Node inputs inside a Pipeline)::

    d, m = Input("d"), Input("m")
    pipe = Pipeline(inputs=[d, m], outputs=[Filter()(d, m)])
    survivors, idx = pipe(data_array, mask_array)
