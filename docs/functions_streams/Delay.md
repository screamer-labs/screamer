---
name: Delay
title: Delay
kind: class
short: Shift each event's index by a fixed duration; values are unchanged.
topics:
- streams
covers:
- delay
---

# `Delay`

Stateless index re-stamp. Each event is forwarded with its index shifted by
`duration` (in index units); the values array is passed through unchanged. The
operation is lossless, 1:1, and order-preserving (a positive constant shift
keeps events monotonic). No warmup, no state.

An explicit index is required. Calling `Delay(duration)(values)` without an
index raises `TypeError`. Usable eagerly (raw `(values, index)` tuples) and
inside a `Pipeline`.

<!-- HELP_END -->

## Example

Shift each event's index forward by 5 units.

```{eval-rst}
.. exec_code::

   import numpy as np
   from screamer import Delay

   vals = np.array([1.0, 2.0, 3.0])
   idx  = np.array([0, 7, 14], dtype=np.int64)

   v, i = Delay(5)(vals, idx)
   print(v)
   print(i)
```
