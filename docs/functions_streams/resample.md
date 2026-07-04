# `resample`

Causal windowed downsampling. Group a stream into fixed index-interval buckets
(`every=`) or fixed event-count buckets (`count=`), and reduce each bucket with
one of `first`, `last`, `min`, `max`, `sum`, `count`, `mean`, or `ohlc`. A
bucket emits only once a later index proves it complete, and the trailing partial
bucket emits at the end of the input. Usable eagerly and inside a `Dag`.

```{eval-rst}
.. autofunction:: screamer.streams.resample
```

## Example

Downsample tick values into buckets of width 10 and take the mean of each bucket.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import resample
   # --- hide: stop ---
   idx  = np.array([0, 3, 10, 12, 20])
   vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

   bar_means, bar_idx = resample(vals, index=idx, every=10, agg="mean")
   print(bar_means)
   print(bar_idx)
```

Use `agg="ohlc"` for an open/high/low/close bar (a four-column result), or
`count=` to bucket by a fixed number of events instead of an index interval.
