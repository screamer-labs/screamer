# `pace`

Replay a stored `(keys, values)` series as an async event stream. `speed=inf`
runs a max-speed backtest; a finite `speed` replays in wall-clock time, turning
key deltas into sleep durations. The values and their order are identical to the
batch and streaming forms; only the timing differs.

```{eval-rst}
.. autofunction:: screamer.streams.pace
```

## Example

A max-speed backtest yields each `(key, value)` event in order, with no delay.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import asyncio
   import numpy as np
   from screamer.streams import pace
   # --- hide: stop ---
   async def backtest():
       series = (np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]))
       return [(e[0], e[1]) async for e in pace(series, speed=float("inf"))]

   print(asyncio.run(backtest()))
```
