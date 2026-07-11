---
name: CombineLatest
title: CombineLatest
kind: class
short: "As-of latest-value join of N streams: one row per distinct index (same-index events coalesce)."
topics:
- streams
covers:
- combine_latest
---

# `CombineLatest`

An as-of join for streams that tick on different clocks. It aligns N streams by
carrying each input's most recent value forward and emitting **one row per
distinct index** - same-index events from multiple streams coalesce into a
single settled row. `emit="when_all"` (default) waits until every input is warm;
`emit="on_any"` starts from the first event.

<!-- HELP_END -->

Feeding `CombineLatest` lazy iterators of `(value, index)` events returns a
lazy iterator (a no-index source is numbered by a per-source arrival counter);
feeding arrays or `(values, index)` tuples returns the eager `(aligned, index)` pair.

## Example

Two streams, `a` and `b`, tick at different indices. `CombineLatest` produces
one aligned row per distinct index, coalescing the two events that share
index 4 into a single row.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import CombineLatest
   # --- hide: stop ---
   a_v = np.array([10.0, 20.0, 40.0])
   a_k = np.array([1, 2, 4])
   b_v = np.array([1.0, 3.0, 4.0])
   b_k = np.array([1, 3, 4])

   aligned, idx = CombineLatest()(a_v, b_v, index=[a_k, b_k])
   print(idx)
   print(aligned)
```

Index 4 has one event from each stream; they coalesce into a single row
`[40.0, 4.0]` instead of two. The aligned columns feed any single-stream
functor, e.g. `RollingCorr(20)(aligned[:, 0], aligned[:, 1])`.
