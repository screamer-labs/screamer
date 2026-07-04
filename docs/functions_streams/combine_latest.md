# `combine_latest`

An as-of join for streams that tick on different clocks. It emits an aligned row
whenever any input advances, carrying each input's most recent value forward
(forward-fill). `emit="when_all"` (default) waits until every input is warm;
`emit="on_any"` starts from the first event.

```{eval-rst}
.. autofunction:: screamer.streams.combine_latest
```

## Example

Two feeds, `a` and `b`, tick at different keys. `combine_latest` produces one
aligned row per advancing key, each column holding that feed's latest value.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import combine_latest
   # --- hide: stop ---
   a = (np.array([1, 2, 4]), np.array([10.0, 11.0, 13.0]))
   b = (np.array([1, 3, 4]), np.array([5.0, 7.0, 9.0]))

   keys, aligned = combine_latest(a, b)
   print(aligned)
```

The aligned columns feed any single-series functor, e.g.
`RollingCorr(20)(aligned[:, 0], aligned[:, 1])`.
