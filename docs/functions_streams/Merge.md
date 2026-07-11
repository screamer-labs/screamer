---
name: Merge
title: Merge
kind: class
short: Merge N value streams into one index-sorted (values, sources, index).
topics:
- streams
covers:
- merge
---

# `Merge`

Interleave several streams into one index-sorted stream, tagging each event with
the integer index of the source it came from. This is the causal, order-preserving
way to fan several streams into a single timeline.

<!-- HELP_END -->

Feeding `Merge` lazy iterators of `(value, index, source)` events returns a lazy
iterator (each source is numbered by a per-source arrival counter); feeding
arrays or `Stream` objects returns the eager `(values, sources, index)` 3-tuple
(Rule A).

## Example

Two streams are merged into one stream ordered by index. `sources` records which
stream each event came from.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Merge
   # --- hide: stop ---
   a_v = np.array([1.0, 3.0])
   a_k = np.array([1, 3])
   b_v = np.array([2.0, 4.0])
   b_k = np.array([2, 4])

   values, sources, idx = Merge()(a_v, b_v, index=[a_k, b_k])
   print(list(zip(idx.tolist(), sources.tolist())))
```

`split` is the inverse: it restores the original per-source streams.
