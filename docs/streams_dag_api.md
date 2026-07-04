# Streams & DAG API reference

The `screamer.streams` combinators align, shape, resample, and replay streams
that do **not** tick together; the `screamer.dag` layer (`Input`, `Dag`) wires
functors and combinators into a graph you define once and run in batch or live.
For the concepts behind keys, alignment, and causality, read
[Streams, keys, and alignment](multistream.md) first.

Every combinator has an eager array form and, where applicable, a streaming twin
(`*_iter`) that yields events one at a time. Signatures and docstrings below are
pulled directly from the code, so they always match the installed version.

A **stream** is a `(keys, values)` pair — `keys` an `int64`/`float64` ordering
(timestamp, tick counter, or row number) and `values` a 1-D or 2-D array. Inside
a `Dag`, a whole stream is a single `Node`.

## Aligning streams

`combine_latest` is the as-of join: it emits an aligned row whenever any input
advances, carrying each input's most recent (forward-filled) value. `merge`
interleaves streams into one key-sorted, source-tagged stream.

```{eval-rst}
.. autofunction:: screamer.streams.combine_latest
.. autofunction:: screamer.streams.merge
```

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import combine_latest
   # --- hide: stop ---
   # two feeds on different clocks: a ticks at keys 1,2,4 ; b at 1,3,4
   keys, aligned = combine_latest(
       (np.array([1, 2, 4]), np.array([10.0, 11.0, 13.0])),
       (np.array([1, 3, 4]), np.array([5.0, 7.0, 9.0])),
   )
   print("keys   :", keys)
   print("a col  :", aligned[:, 0])   # a's latest at each aligned key
   print("b col  :", aligned[:, 1])   # b's latest (forward-filled)
```

The two aligned columns feed any single-series functor unchanged, e.g.
`RollingCorr(20)(aligned[:, 0], aligned[:, 1])`. (Inside a `Dag` the shorter
`RollingCorr(20)(combine_latest(a, b))` form works, because there
`combine_latest(a, b)` is a `Node`.)

## Shaping streams

`dropna` removes events whose value is `NaN`; `select` projects column(s) out of
a wide stream; `filter` keeps events a predicate accepts; `split` inverts
`merge`. `dropna` and `select` are also usable inside a `Dag`; `filter` (a Python
predicate) and `split` are eager-only.

```{eval-rst}
.. autofunction:: screamer.streams.dropna
.. autofunction:: screamer.streams.select
.. autofunction:: screamer.streams.filter
.. autofunction:: screamer.streams.split
```

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import dropna, select
   # --- hide: stop ---
   k, v = dropna(np.array([1, 2, 3]), np.array([1.0, np.nan, 3.0]))
   print("dropna keys:", k, "values:", v)

   wide = np.array([[10.0, 11.0], [20.0, 21.0]])
   _, col1 = select(np.array([1, 2]), wide, 1)   # keep column 1
   print("select col1:", col1)
```

## Resampling

`resample` is causal windowed downsampling — fixed key-interval (`width=`) or
fixed event-count (`count=`) buckets, reduced with `first/last/min/max/sum/
count/mean/ohlc`. It is fully causal: a bucket emits only once a later key proves
it complete, and the trailing partial bucket emits at end of input. `agg="ohlc"`
produces a width-4 stream (`open, high, low, close`).

```{eval-rst}
.. autofunction:: screamer.streams.resample
.. autofunction:: screamer.streams.resample_iter
```

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer.streams import resample
   # --- hide: stop ---
   keys = np.array([0, 3, 10, 12, 20])     # tick keys
   vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
   bar_keys, ohlc = resample(keys, vals, width=10, agg="ohlc")   # 10-key bars
   print("bar keys:", bar_keys)            # left label = bucket start
   print("open high low close:")
   print(ohlc)
```

For multiple streams, resample each to a common `width`/`origin` grid, then
`combine_latest` them — count-based buckets do not align across streams.

## Replay

`pace` turns stored `(keys, values)` series into an async event stream.
`speed=float("inf")` is a max-speed backtest; a finite `speed` replays in
wall-clock time (key deltas become sleep durations). Values and order are
identical to the batch/streaming forms — only *timing* differs.

```{eval-rst}
.. autofunction:: screamer.streams.pace
```

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import asyncio
   import numpy as np
   from screamer.streams import pace
   # --- hide: stop ---
   async def backtest():
       out = []
       async for e in pace((np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0])),
                           speed=float("inf")):
           out.append((e[0], e[1]))   # (key, value)
       return out

   print(asyncio.run(backtest()))
```

## Computational DAG

`Input` names a source stream; calling functors and combinators on `Input`/`Node`
handles records a graph. `Dag(inputs=[...], outputs=[...])` compiles it into a
callable that runs the **same** graph in batch (`dag(...)`) or live
(`dag.stream(...)`) with byte-identical results.

```{eval-rst}
.. autoclass:: screamer.dag.Input
.. autoclass:: screamer.dag.Dag
   :members: stream
```

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Input, Dag, Sub
   from screamer.streams import combine_latest
   # --- hide: stop ---
   a, b = Input("a"), Input("b")
   spread = Sub()(combine_latest(a, b))          # align, then difference
   dag = Dag(inputs=[a, b], outputs=[spread])

   fa = (np.array([1, 2, 3]), np.array([10.0, 20.0, 30.0]))
   fb = (np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]))
   batch_k, batch_v = dag(fa, fb)                # run in batch
   live_k, live_v = dag.stream(fa, fb)           # run live, event by event
   print("spread     :", batch_v.reshape(-1))
   print("batch==live:", np.array_equal(batch_v, live_v))
```

## See also

- [Streams, keys, and alignment](multistream.md) — the concepts (order keys,
  the alignment layer, causality, batch == streaming).
- The **Examples** notebooks (07–10) walk through aligning async feeds, replay,
  stream shaping, and the DAG end to end.
