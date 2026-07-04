# `dropna`

Drop events whose value is `NaN`. This changes the length of the stream (it is a
cardinality-changing combinator, unlike the shape-preserving compute functors).
Usable eagerly and inside a `Dag`.

```{eval-rst}
.. autofunction:: screamer.streams.dropna
```

## Example

The event at key 2 is `NaN`, so it is removed; the surviving values are returned.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import dropna
   # --- hide: stop ---
   keys, values = dropna(np.array([1, 2, 3]), np.array([1.0, np.nan, 3.0]))
   print(values)
```
