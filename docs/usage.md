# Using screamer

screamer builds a calculation once and runs it on stored data or a live stream
without changes. Each section below links to a runnable notebook and a reference
page for deeper treatment.

## Constructing and calling a function

Some calculations look at each value in isolation. `abs(x)` is one: the result
at every step depends only on the input at that step. Others need memory. A mean
over the last three samples cannot produce its value at a given step without the
two samples before it, so it has to carry them forward. That carried-forward
data is the object's state, and it is why screamer splits construction from the
call. Constructing the object sets up its state; calling it feeds in data and
advances that state.

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

The first two outputs are `NaN` because the window of size 3 is not full yet;
see [Warmup](#warmup) below. Because the object carries state between calls,
feeding it values one at a time accumulates history.

## Stored data and live streams, one calculation

When you pass an array or a list, screamer has the whole dataset in hand and processes it in one go. When you pass an iterator, it works as a live stream, taking one value at a time as you pull from it, which is what an event loop needs. A single scalar is just a stream of length one. The output type always mirrors the input.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   # --- hide: stop ---
   print("scalar:", RollingMean(3)(2.0))                          # -> a float
   print("array :", RollingMean(3)(np.array([1., 2., 3., 4., 5.])))  # -> an array
   print("list  :", RollingMean(3)([1., 2., 3., 4., 5.]))         # -> a list
   print("stream:", list(RollingMean(3)(iter([1., 2., 3., 4., 5.]))))  # -> lazy
```

Every function is **causal**: its output at each step depends only on current
and past inputs, never future ones. That is why code developed on stored
history can be deployed against a live feed without changes. The exact
input-to-output contract for every type is in the
[Polymorphic API reference](polymorphic_api.md). A worked version of this
example is in the [quickstart notebook](notebooks/01-quickstart-polymorphic-api),
and the [streaming notebook](notebooks/06-streaming-live-events) shows the live
path on a longer example.

## Processing several series

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

Most functions need a few samples before they can produce a defined value: a
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
nesting.

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

Some functions take more than one series. You can pass the series in either of
two shapes, whichever your data is already in: as separate arguments, one per
series, or as the columns of a single 2-D array. The two forms are equivalent
and return the same result.

`RollingCorr`, the rolling correlation of two series, accepts both:

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingCorr
   # --- hide: stop ---
   x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
   y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])   # y = 2x, so correlation is 1.0 once warm
   xy = np.column_stack([x, y])               # x and y as the two columns of one array

   print("separate:", RollingCorr(window_size=3)(x, y))
   print("packed  :", RollingCorr(window_size=3)(xy))
```

Each argument follows the single-input rules, so the output type mirrors the
input type. Price indicators built on open, high, low, and close (OHLC) bars take
their series the same way: `ATR(14)(high, low, close)` and `ATR(14)(hlc)`, a
`(T, 3)` array, read identically. The
[financial-indicators notebook](notebooks/03-financial-indicators) works through
them, and the [Polymorphic API reference](polymorphic_api.md) has the exact
contract.

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

Real streams have gaps: a missing tick, a sensor dropout, a `NaN` left by an
upstream function's warmup. Every screamer function declares exactly how it
responds to a `NaN` input, following one of three policies (`ignore`,
`propagate`, `nan-aware`). A `NaN` never corrupts internal state, and the
function always recovers. The full contract, and the `FillNa` / `Ffill`
functions for cleaning gaps, are in [NaN and warmup](nan_and_warmup.md); the
[NaN-handling notebook](notebooks/05-nan-handling) demonstrates each policy.

## Aligning streams

The examples above assume inputs are aligned: row `i` of one pairs with row `i`
of another. When feeds arrive on different clocks, at different rates, or with
dropped samples, screamer's stream operators align them before they reach a
function. They combine streams (`CombineLatest`, `Merge`), reshape them
(`Dropna`, `Filter`, `Select`), and downsample them (`Resample`).
The model (streams, their index, and alignment) is described in
[Streams, values, and alignment](multistream.md). The
[Multi-stream operators](notebooks/07-multi-stream-operators) notebook shows them
in use.

## Whole pipelines

A `Pipeline` wires functions and stream operators into a reusable unit you define
once and call. You name your sources with `Input`, apply
functions to them, and the code reads top-to-bottom like an ordinary script.
Building a `Pipeline` records the wiring; nothing runs until you call it.

```python
from screamer import Input, Pipeline, RollingMean, Sub, CombineLatest

a, b = Input("a"), Input("b")         # two named sources
spread = Sub()(CombineLatest()(a, b)) # align the two sources, then subtract
signal = RollingMean(10)(spread)      # smooth the spread

pipe = Pipeline(inputs=[a, b], outputs=[signal])   # build the pipeline

# pipe(arr_a, arr_b)          -> run on stored arrays
# pipe(gen_a, gen_b)          -> feed generators to run live, event by event
```

Calling a `Pipeline` runs every step its outputs depend on. The model and its
guarantees are in [Pipelines](pipelines.md); the
[Pipelines notebook](notebooks/08-pipelines) builds and runs a complete one.

## Resetting state

Because an object carries state, it matters when that state clears. screamer
resets it for you on the stored-data paths: at the start and end of every array
or list call, and between the columns of a multi-column array. So passing an
array gives the same result as constructing a fresh object for it, and columns
never influence each other.

On the streaming paths (feeding scalars or an iterator), state is deliberately
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

- **Worked examples**: the [example notebooks](notebooks/01-quickstart-polymorphic-api)
  cover statistics, financial indicators, signal processing, async streams, and
  pipelines, each self-contained and runnable.
- **Concepts and contracts**: [Polymorphic API](polymorphic_api.md),
  [NaN and warmup](nan_and_warmup.md),
  [Streams, values, and alignment](multistream.md), and
  [Pipelines](pipelines.md) give the precise behaviour.
- **The function catalog**: browse every function by family or by use case in
  the reference sections.
