# `merge`

Interleave several streams into one key-sorted stream, tagging each event with
the index of the source it came from. This is the causal, order-preserving way to
fan several feeds into a single timeline.

```{eval-rst}
.. autofunction:: screamer.streams.merge
```

## Example

Two feeds are merged into one stream ordered by key. `sources` records which feed
each event came from.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import merge
   # --- hide: stop ---
   keys, values, sources = merge(
       (np.array([1, 3]), np.array([1.0, 3.0])),
       (np.array([2, 4]), np.array([2.0, 4.0])),
   )
   print(list(zip(keys.tolist(), sources.tolist())))
```

`split` is the inverse: it restores the original per-source streams.
