---
name: CombineLatest
title: CombineLatest
kind: class
short: 'As-of latest-value join of N streams: one row per distinct index (same-index events
  coalesce).'
tags:
- as-of join
- align
- join
- combine latest
- synchronize
see_also:
- Merge
- Resample
- Select
- split
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

## Examples

Two streams, `a` and `b`, tick at different indices. `CombineLatest` produces
one aligned row per distinct index, coalescing the two events that share
index 4 into a single row.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import CombineLatest

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

### Multi-column streams and `layout`

The columns of each input are fields, and `CombineLatest` concatenates them: a
`(T, 4)` bar stream and a `(T,)` signal give `(T, 5)` rows. When the columns
are assets instead, `layout=` names the axes of each input with `T` (time), `A`
(asset), and `F` (field). The asset axes broadcast, so one market series is
paired with every asset of a panel. An asset layout runs on arrays, `(values,
index)` tuples, and iterators of events, and raises `ValueError` inside a
`Pipeline`.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import CombineLatest

   bars = np.arange(12.0).reshape(3, 4)        # 3 bars of [open, high, low, close]
   signal = np.array([1.0, -1.0, 1.0])
   rows, _ = CombineLatest()(bars, signal)
   print(rows.shape)

   panel = np.arange(6.0).reshape(3, 2)        # 2 assets
   market = np.array([10.0, 20.0, 30.0])
   pairs, _ = CombineLatest(layout=("TA", "T"))(panel, market)
   print(pairs.shape)
   print(pairs[:, 1])                          # asset 1 next to the market
