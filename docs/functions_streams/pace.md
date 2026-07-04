# `pace`

Replay stored value streams as an async event stream. `speed=inf` runs a
max-speed backtest; a finite `speed` replays in wall-clock time, turning index
deltas into sleep durations. The values and their order are identical to the
batch and streaming forms; only the timing differs.

```{eval-rst}
.. autofunction:: screamer.streams.pace
```

## Example

A max-speed backtest yields each `(value, index, source)` event in order, with
no delay.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import asyncio
   import numpy as np
   from screamer.streams import pace
   # --- hide: stop ---
   async def backtest():
       vals = np.array([1.0, 2.0, 3.0])
       idx  = np.array([0, 1, 2], dtype=np.int64)
       return [(v, i, s) async for v, i, s in pace(vals, index=[idx], speed=float("inf"))]

   print(asyncio.run(backtest()))
```
