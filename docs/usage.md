# Using screamer

This guide walks through screamer's core ideas in the order they build on each
other: how a function is called, why the same call works on stored data and on a
live stream, how data shape is handled, and how functions compose into larger
pipelines. Each section is short and links to a runnable notebook or a reference
page where you can go deeper.

## Construct, then call

Every screamer function is used in two steps. First you create an **object**,
configured with its parameters. Then you call that object on your data.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   # --- hide: stop ---
   ma = RollingMean(window_size=3)      # 1. construct, with parameters
   result = ma(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))   # 2. call, on data
   print(result)
```

The object holds the algorithm's state, so you can keep a reference to it and
call it many times. The first two outputs are `NaN` because the window of size
3 is not full yet; see [Warmup](#warmup) below.

## One object, any input

The same object accepts a single value, a NumPy array, a Python list, or an
iterator, and the type of the output mirrors the type of the input. This is the
central idea in screamer: you write a calculation once and use it on a single
number, a stored array, or a live feed, without changing anything.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   # --- hide: stop ---
   print("scalar:", RollingMean(3)(2.0))                          # -> a float
   print("array :", RollingMean(3)(np.array([1., 2., 3., 4., 5.])))  # -> an array
   print("list  :", RollingMean(3)([1., 2., 3., 4., 5.]))         # -> an array
   print("stream:", list(RollingMean(3)(iter([1., 2., 3., 4., 5.]))))  # -> lazy
```

An array or list is processed eagerly, in one pass. An iterator is processed
lazily: results come out one at a time as you advance it, which is what a live
event loop needs. The exact input-to-output contract for every type is in the
[Polymorphic API reference](polymorphic_api.md). A worked version of this
example is in the [quickstart notebook](notebooks/01-quickstart-polymorphic-api).

## Batch equals streaming

Because the same object runs both ways, the numbers it produces are identical
whether you pass a whole array at once or feed values one at a time. You can
develop and test on stored history, then run the exact same code live.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   # --- hide: stop ---
   data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

   batch = RollingMean(3)(data)              # whole array at once

   live = RollingMean(3)
   streamed = [live(x) for x in data]        # one value at a time

   print("batch   :", batch)
   print("streamed:", np.array(streamed))
   print("identical:", np.allclose(batch, streamed, equal_nan=True))
```

This holds because every function is **causal**: its output at each step depends
only on the current and past inputs, never on future ones. There is no
look-ahead, so a value computed live matches the one computed in a backtest. The
[streaming notebook](notebooks/06-streaming-live-events) shows this on a longer
example.

## Many series at once

For a 2-D array, screamer treats the **first axis as time** and every other axis
as a parallel, independent series. A `(T, K)` array is `K` series of length `T`,
each processed on its own.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   # --- hide: stop ---
   data = np.array([[1.0, 10.0],
                    [2.0, 20.0],
                    [3.0, 30.0],
                    [4.0, 40.0]])

   print(RollingMean(3)(data))     # shape (4, 2): the two columns are independent
```

The state of one column never leaks into another. There is no `axis=` argument
to set, because axis 0 is always time; if you need the operation along a
different axis, transpose the array before passing it. The full convention,
including higher-dimensional arrays, is in the
[Polymorphic API reference](polymorphic_api.md).

## Warmup

Most functions need a few samples before they can produce a defined value — a
mean over a window of 20 has nothing to report until 20 samples have arrived.
During this warmup they emit `NaN`. The `start_policy` argument controls the
behaviour: the default `"strict"` waits for a full window, while `"expanding"`
produces a value from the first sample using whatever data is available so far.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   # --- hide: stop ---
   data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

   print("strict   :", RollingMean(3)(data))
   print("expanding:", RollingMean(3, start_policy="expanding")(data))
```

The full definition of each policy, and how warmup interacts with missing data,
is in [NaN and warmup](nan_and_warmup.md).

## Chaining functions

Every function accepts the output of another, so calculations compose by
nesting. A chain runs identically in batch and streaming, exactly like a single
function.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMax, Diff
   # --- hide: stop ---
   data = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0])

   diff = Diff(1)                      # step-to-step change
   rmax = RollingMax(window_size=3)    # rolling maximum of that change
   print(rmax(diff(data)))
```

## Functions with several inputs

Some functions take more than one series. `RollingCorr`, for example, is the
rolling correlation of two inputs. The call shape mirrors the single-input case:
two scalars give a scalar, two arrays give an array, two streams give a stream.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingCorr
   # --- hide: stop ---
   x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
   y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

   print(RollingCorr(window_size=3)(x, y))   # y = 2x, so correlation is 1.0 once warm
```

The multi-input contract, including the `(T, N)` array form used for OHLC
indicators, is in the [Polymorphic API reference](polymorphic_api.md), and the
[financial-indicators notebook](notebooks/03-financial-indicators) puts it to
work.

## Functions with several outputs

Some functions produce more than one value per step. `BollingerBands` returns a
lower, middle, and upper band. The output gains a trailing axis of that size.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import BollingerBands
   # --- hide: stop ---
   data = np.random.default_rng(0).standard_normal(100)

   bands = BollingerBands(20)(data)
   print("shape:", bands.shape)     # (100, 3): lower, mid, upper
```

## Missing data

Real streams have gaps — a missing tick, a sensor dropout, a `NaN` left by an
upstream function's warmup. Every screamer function declares exactly how it
responds to a `NaN` input, following one of three policies (`ignore`,
`propagate`, `nan-aware`). A `NaN` never corrupts internal state, and the
function always recovers. The full contract, and the `FillNa` / `Ffill`
functions for cleaning gaps, are in [NaN and warmup](nan_and_warmup.md); the
[NaN-handling notebook](notebooks/05-nan-handling) demonstrates each policy.

## Streams that don't tick together

The examples above assume inputs are aligned — row `i` of one pairs with row `i`
of another. When feeds arrive on different clocks, at different rates, or with
dropped samples, the `screamer.streams` layer aligns them before they reach a
function. It provides operators to combine streams (`combine_latest`, `merge`),
reshape them (`dropna`, `filter`, `select`), downsample them (`resample`), and
replay stored data as a live feed (`replay`). The model — streams, their index,
and alignment — is described in
[Streams, values, and alignment](multistream.md), with the [`Stream`](functions_streams/Stream.md)
type as its object form. The [Working with streams](notebooks/07-working-with-streams)
notebook shows them in use.

## Whole pipelines: the graph

A `Dag` lets you wire functions and stream operators into a **computational
dependency graph** you define once and run either on stored data or live, with
identical results. You build it by naming your sources with `Input` and applying
functions to them; the code reads top-to-bottom like an ordinary script, but
what you are describing is a graph of dependencies. Building it only records the
structure — nothing computes until you call the compiled graph, at which point
the engine evaluates each node in dependency order.

```python
from screamer import Input, Dag, RollingMean, Sub
from screamer.streams import combine_latest

a, b = Input("a"), Input("b")         # two named sources
spread = Sub()(combine_latest(a, b))  # a node: align the two sources, then subtract
signal = RollingMean(10)(spread)      # a node that depends on `spread`

dag = Dag(inputs=[a, b], outputs=[signal])   # compile the graph

# dag(feed_a, feed_b)        -> run on stored arrays
# dag.stream(feed_a, feed_b) -> run live, event by event, with identical results
```

Each operation is a node whose parents are its inputs, and calling the graph
evaluates every node the requested outputs depend on. The model and its
guarantees are in [The computational graph](dag.md); the
[DAG notebook](notebooks/09-computational-dag) builds and runs a complete one.

## Resetting state

Because an object carries state, it matters when that state clears. screamer
resets it for you on the batch paths: at the start and end of every array or
list call, and between the columns of a multi-column array. So passing an array
gives the same result as constructing a fresh object for it, and columns never
influence each other.

On the streaming paths — feeding scalars or an iterator — state is deliberately
**not** reset, because preserving it across calls is the whole point of
streaming. To start over mid-stream, call `reset()` or build a new object.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   # --- hide: stop ---
   ma = RollingMean(window_size=3)
   ma(1.0)
   ma(2.0)          # the object now holds two samples of state
   ma.reset()       # back to empty; the next call starts a fresh window
   print("reset done")
```

The precise reset rules are in the [Polymorphic API reference](polymorphic_api.md).

## Where to go next

- **Worked examples** — the [example notebooks](notebooks/01-quickstart-polymorphic-api)
  cover statistics, financial indicators, signal processing, async streams, and
  the DAG, each self-contained and runnable.
- **Concepts and contracts** — [Polymorphic API](polymorphic_api.md),
  [NaN and warmup](nan_and_warmup.md),
  [Streams, values, and alignment](multistream.md), and
  [The computational graph](dag.md) give the precise behaviour.
- **The function catalog** — browse every function by family or by use case in
  the reference sections.
