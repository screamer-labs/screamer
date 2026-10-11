# Streams in depth

An operator runs on a live feed in two ways: one number per call, `op(x)`, or
one call on an iterator or async iterator that yields the events. This page
describes what each form does with the operator's state, when it reads
its sources, how several sources pair up, what an event may hold, and how
errors behave. The examples are in Python. JavaScript behaves the same, apart
from the differences listed under
[Parity gaps](javascript.md#parity-gaps).

## The running series and independent runs

An operator holds one **running series**: the state that `op(x)` reads and
advances. A call with one number is one step of that series, so successive
calls continue it.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   from screamer import RollingMean
   # --- hide: stop ---
   mean = RollingMean(3)
   print([mean(x) for x in [1.0, 2.0, 3.0, 4.0]])
```

Every other call is an **independent run**: an array, a list, an iterator, or
an async iterator. An independent run starts from a clean state, gives every
asset its own state, never reads or advances the running series, and gives the
rows of the batch call on the same data. A stream runs on copies of the
operator made from its parameters, so two streams of one operator never share
state, whether they run one after the other or interleaved step by step:

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   from screamer import RollingMean
   # --- hide: stop ---
   mean = RollingMean(3)
   a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
   b = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]

   # one after the other
   print(list(mean(iter(a))))
   print(list(mean(iter(b))))

   # interleaved: one step of each stream in turn
   stream_a, stream_b = mean(iter(a)), mean(iter(b))
   for x, y in zip(stream_a, stream_b):
       print(x, y)
```

A batch call and the opening of a stream both reset the running series. The
next `op(x)` starts a new warmup, and later live calls do not touch a stream
that is already open:

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   # --- hide: stop ---
   mean = RollingMean(3)
   print([mean(x) for x in [1.0, 2.0, 3.0]])   # the running series is warm
   mean(np.array([10.0, 20.0, 30.0]))          # a batch call resets it
   print(mean(4.0))                            # NaN: a new warmup

   mean = RollingMean(3)
   print([mean(x) for x in [1.0, 2.0, 3.0]])
   stream = mean(iter([100.0, 200.0, 300.0]))  # opening a stream resets it too
   print([mean(x) for x in [4.0, 5.0, 6.0]])   # a new warmup
   print(list(stream))                         # the stream is unaffected
```

To continue the running series from an iterator, as a stream did in Python
2.8.0, call the operator once per value: `(mean(x) for x in source)`.

Because a stream copies the operator from its parameters, it needs an operator
constructed from the `screamer` namespace (in JavaScript, from its factory).
Streaming an instance of a raw binding class raises `TypeError`.

## Laziness

A stream reads nothing from its sources before the first `next()`, and then
reads one event per step. A generator that blocks on a socket or a queue is
therefore read at the pace the consumer asks for results.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   from screamer import RollingMean
   # --- hide: stop ---
   def source():
       for x in [1.0, 2.0, 3.0]:
           print("read", x)
           yield x

   stream = RollingMean(2)(source())
   print("opened")
   print(next(stream))
   print(next(stream))
```

Arrays, lists and tuples are processed eagerly in one pass. A list is the eager
form of a stream: `op([e1, e2, ...])` returns the list of results the iterator
`op(iter([e1, e2, ...]))` would yield.

## Several sources, and async iterables

A multi-input operator takes one source per input, or one source whose events
hold every input. Each step reads one event from every source, and the stream
ends when the shortest source ends. An async iterable (an `async def`
generator, or any object with `__aiter__`) gives an async iterator that awaits
one event from every source per step:

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import asyncio
   from screamer import Mul
   # --- hide: stop ---
   print(list(Mul()(iter([1.0, 2.0, 3.0]), iter([10.0, 20.0]))))
   print(list(Mul()(iter([(1.0, 10.0), (2.0, 20.0)]))))   # one source of pairs

   async def ticks(values):
       for v in values:
           await asyncio.sleep(0)
           yield v

   async def main():
       stream = Mul()(ticks([1.0, 2.0, 3.0]), ticks([10.0, 20.0, 30.0, 40.0]))
       print([y async for y in stream])

   asyncio.run(main())
```

All sources of one call are synchronous, or all are async; mixing the two
raises `TypeError`. An async stream runs one step at a time: in Python, awaiting a second
`__anext__()` while the first is still running raises `RuntimeError`.
JavaScript queues such calls and answers them in order, as an async generator
does.

## Errors during a stream

In Python, an event that fails, such as one with the wrong shape, raises from
that `next()` call. The stream stays open and the following call reads the next
event. An exception raised by a source propagates from `next()`; a generator
that raised has finished, so the stream then ends. Python closes nothing on an
error: the sources are released when the stream is garbage collected.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   # --- hide: stop ---
   events = [np.zeros(3), np.zeros(2), np.ones(3)]
   stream = RollingMean(2)(iter(events))
   print(next(stream))
   try:
       next(stream)
   except ValueError as error:
       print(error)
   print(next(stream))
```

In JavaScript, an error in an event, in `throw()` or in `return()` closes every
source and ends the stream, as a generator does; the next `next()` returns
`{ done: true }`. A JavaScript stream dropped before its end frees its operator copies
when it is garbage collected; `return()` or the operator's `dispose()` frees
them at once.

Both conventions are intentional. Each follows the native behavior of its
language: a Python stream behaves like `map`, a JavaScript stream like a
generator. To release the sources of a Python stream at a known point, close
them yourself, for example with `contextlib.closing`:

```python
from contextlib import closing

with closing(source_generator()) as src:
    for y in RollingMean(3)(src):
        ...
```

## Events: numbers, asset arrays, and packed fields

An event is one time step of shape `(*A, F?)` (see
[Time, assets and fields](polymorphic_api.md#time-assets-and-fields)). A
number is one value. An array of shape `(K,)` holds one value for each of `K`
assets, and each asset runs on its own copy of the operator:

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   # --- hide: stop ---
   X = np.arange(12.0).reshape(4, 3)          # 4 steps of 3 assets
   for row in RollingMean(2)(iter(X)):
       print(row)
```

For a multi-input operator, a packed event of shape `(K, N)` holds the `N`
inputs of each asset along its last axis. Separate sources broadcast like
separate arrays in a batch call, so a source of numbers pairs with every asset
of a source of `(K,)` events:

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import Mul
   # --- hide: stop ---
   packed = [np.array([[1.0, 10.0], [2.0, 20.0]]),
             np.array([[3.0, 30.0], [4.0, 40.0]])]
   print(list(Mul()(iter(packed))))

   scale = iter([1.0, 2.0])
   rows = iter([np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])])
   print(list(Mul()(scale, rows)))
```

The first event fixes the shape of the stream: numbers, or arrays of one shape.
A later event of another shape raises `ValueError` ("every event must have the
same shape"). An event of shape `(1,)` is one asset, not a number.

A one-input operator cannot take a tuple of arrays. A `(values, index)` pair,
the form stream operators such as `Resample` and `CombineLatest` return, raises
`TypeError` there rather than being read as two events. Pass the values and keep
the index as it is, or build a [Pipeline](pipelines.md), which carries the index
from node to node:

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   # --- hide: stop ---
   values = np.array([1.0, 2.0, 3.0, 4.0])
   index = np.array([10, 20, 30, 40])
   try:
       RollingMean(2)((values, index))
   except TypeError as error:
       print(error)
   result = (RollingMean(2)(values), index)
   print(result)
```

## Row operators and portfolio engines

A row operator reads all assets of a step together: the cross-sectional
operators such as `CrossRank`, and the portfolio engines. It streams one `(K,)`
row per input, from one source per input or from one source of tuples of rows.
A row operator reads a tuple or list of arrays as rows over time, so unpack a
`(values, index)` pair first; it never needs the index.
A portfolio engine yields one `(K, 4)` row of `[equity, pnl, position, cost]`
per step, and `PortfolioReport` streams those rows into the book's report:

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import CrossRank, BacktestPortfolioTarget, PortfolioReport
   # --- hide: stop ---
   scores = np.array([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
   print(list(CrossRank()(iter(scores))))

   prices = np.array([[100.0, 50.0], [101.0, 49.0], [102.0, 51.0], [103.0, 52.0]])
   weights = np.array([[0.5, -0.5], [np.nan, np.nan], [0.5, 0.5], [np.nan, np.nan]])
   engine_rows = list(BacktestPortfolioTarget()(iter(weights), iter(prices)))
   print(engine_rows[-1])
   report = list(PortfolioReport()(iter(engine_rows)))
   names = ["drawdown", "cum_cost", "turnover", "trades", "max_drawdown", "sharpe"]
   print({name: round(x, 4) for name, x in zip(names, report[-1])})
```

## One stream of rows, or one operator per asset

To run one operator on `K` assets live, prefer one stream of `(K,)` rows over
`K` operators called with `op(x)`. The stream makes one call per step, and C++
updates all `K` states in it, where `K` operators make `K` Python calls per
step. Every row holds a value for every asset, and a NaN is passed to that
asset's operator like any other missing value (see
[NaN and warmup](nan_and_warmup.md)). Use one operator per asset when the assets tick at
different times, or align them first with `CombineLatest`.

A `(K,)` array passed directly, `op(row)`, is a batch of `K` time steps of one
series, not one event of `K` assets. Pass the feed as an iterator of `(K,)`
arrays instead: a generator that blocks on a socket or a queue, or
`iter(feed)` for snapshots already in memory:

```python
for means in RollingMean(20)(iter(feed)):   # feed: (K,) arrays of the latest prices
    publish(means)
```
