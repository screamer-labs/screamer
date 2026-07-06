---
name: replay
title: replay
kind: function
short: Replay merged streams as an async event stream paced by index-deltas.
topics:
- streams
---

# `replay`

Replay stored value streams as an async event stream. `speed=inf` runs a
max-speed backtest; a finite `speed` replays in wall-clock time, turning index
deltas into sleep durations. The values and their order are identical to the
batch and streaming forms; only the timing differs.

<!-- HELP_END -->

```{eval-rst}
.. autofunction:: screamer.streams.replay
```

## Example

A max-speed backtest yields each `(value, index, source)` event in order, with
no delay.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import asyncio
   import numpy as np
   from screamer.streams import replay
   # --- hide: stop ---
   async def backtest():
       vals = np.array([1.0, 2.0, 3.0])
       idx  = np.array([0, 1, 2], dtype=np.int64)
       return [(v, i, s) async for v, i, s in replay(vals, index=[idx], speed=float("inf"))]

   print(asyncio.run(backtest()))
```
