# `filter`

Keep only the events a predicate accepts. The predicate is a Python callable, so
`filter` is eager-only; it is not available inside a `Dag` (the graph engine runs
pure C++ functors, no Python callbacks). For the common case of removing `NaN`,
prefer `dropna`.

```{eval-rst}
.. autofunction:: screamer.streams.filter
```

## Example

Keep the positive values.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import filter
   # --- hide: stop ---
   vals = np.array([5.0, -2.0, 8.0])

   pos_vals, pos_idx = filter(vals, lambda v: v > 0)
   print(pos_vals)
```
