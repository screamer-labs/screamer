# Polymorphic input/output behavior

This page is the exhaustive contract for how a screamer function handles each
input type: exactly what you may pass, what you get back, and the dispatch rules
that decide. For a gentle, example-led introduction to the same idea, see
[Using screamer](usage.md); this page is the reference behind it.

The one property the whole contract exists to guarantee: the same object works
on a single scalar, a NumPy array, a strided view, a list, an iterator, or an
async generator, so code written against stored data runs unchanged on a live
stream.

> For combining, splitting, or filtering streams that do **not** tick
> together (different rates, async arrival, missing samples), see
> [Streams, values, and alignment](multistream.md). The lockstep contract on this
> page is the degenerate "no index → row number" case of that model.


## The six rules

Every operator, in Python and JavaScript, follows six rules. The sections below
give the details of each.

1. **Shape `(T, *A, F?)`.** Time is axis 0, then zero or more asset axes, then
   an optional field axis. On an array of rank 3 or more, a trailing field
   axis of size 1 is the single input field: `RollingMean(5)` on a
   `(T, K, 1)` array returns `(T, K)`, and a `(T, 1)` array stays `(T, 1)`.
   Details: [Time, assets and fields](#time-assets-and-fields).
2. **The container never changes the numbers.** An iterator, an async iterator
   or a list of events gives the rows of the batch result:
   `list(RollingMean(5)(iter(X)))` holds the rows of `RollingMean(5)(X)`.
   Details: [Streams in depth](streaming.md). A pandas `Series` or `DataFrame`
   is read as its values and gives a NumPy array, not a pandas object; use
   `to_pandas` to attach an index again. A `str`, `bytes`, `dict` or `set`
   is not data and raises `TypeError` at the call.
3. **One meaning per container.** A number is one step of the operator's
   running series, an array is a batch along time, and a list or an iterator is
   a sequence of events: `RollingMean(3)(2.0)` returns a `float`, and
   `RollingMean(3)([1.0, 2.0])` returns a `list` of two results. Details:
   [the single-input contract](#the-single-input-contract-screamerbase) and
   [Streams in depth](streaming.md).
4. **Inputs are separate arguments, or one packed array, with assets
   broadcast.** `RollingCorr(20)(X, y)` with `X (T, K)` and `y (T,)` gives
   `(T, K)`, and so does one packed `(T, K, 2)` array. Details:
   [the multi-input contract](#the-multi-input-contract-functorbase_-n-1).
5. **Ambiguity raises.** A `(values, index)` tuple passed to
   `RollingMean(5)` raises `TypeError`, and inputs of different lengths along
   time raise `ValueError`. Details:
   [the dispatch order](#the-dispatch-order).
6. **A change to the result of code that worked before needs an explicit
   decision** and an entry under "Changed" in the
   [release notes](changelog.rst), with the way to get the old result.


The contract has two layers:

- **1‑input / 1‑output classes** (`RollingMean`, `EwVar`, `Diff`, `Lag`,
  every `Rolling*`, every `Ew*`, all the math transforms, ...). These inherit
  from `ScreamerBase`. The vast majority of screamer.
- **N‑input / 1‑output classes** (`RollingCorr`, `RollingCov`,
  `RollingBeta`, `RollingSpread`). These inherit from
  `FunctorBase<Derived, N, 1>`.
- **1‑input / M‑output classes** (`RollingMinMax`, `BollingerBands`).
  These inherit from `FunctorBase<Derived, 1, M>`.

The general N‑input / M‑output case is also supported (`Cart2Polar`,
`Polar2Cart`), inheriting from `FunctorBase<Derived, N, M>`. All four
arities follow the same dispatch and eager/lazy rules described below.


## The single-input contract (`ScreamerBase`)

Every 1‑in/1‑out class supports the following input/output shapes:

| You pass... | You get back... | How |
|---|---|---|
| `int`, `float`, `bool` | `float` | one `process_scalar` call |
| `numpy.float32/64`, `numpy.int32/64`, `numpy.uint32/64` | `float` | one `process_scalar` call |
| 0‑D NumPy array | `float` | rank 0, treated as a scalar |
| 1D NumPy array, length 1 | 1D NumPy array, shape `(1,)` | shape preserved; length 1 is still rank 1, not a scalar |
| 1D NumPy array, length ≥ 2 | 1D NumPy array, same shape | one `process_*` pass over the buffer |
| 1D strided NumPy view | 1D NumPy array, same shape | strided pass; output is contiguous |
| 2D NumPy array `(T, K)` | 2D NumPy array `(T, K)` | one independent stream per column; `reset()` between columns |
| N‑D NumPy array `(T, J, K, ...)` | same shape | `J·K·...` independent streams along axis 0 |
| Python `list` of numbers | Python `list` of same length | container type preserved; processed eagerly in one pass |
| Python `tuple` of numbers | Python `list` of same length | processed eagerly in one pass; the output is a `list` |
| Python `list` of arrays | Python `list` of arrays | a sequence of asset events, one operator state per asset |
| Python `tuple` of arrays | `TypeError` | a `(values, index)` stream goes to a stream operator or a `Pipeline`; a compute operator takes the values. A row operator instead reads it as rows over time |
| Python iterator (`iter(...)`, generator, anything iterable that is not list/tuple/array) | a screamer `LazyEvalIterator` | results yielded **one at a time on demand** |
| Async iterable (an `async def` generator with `yield`, or anything with `__aiter__`) | a screamer `LazyAsyncIterator` | results awaited one at a time; events are numbers or asset arrays, as for an iterator |
| Anything else | `ValueError` | "Unsupported input type for call: ..." |

There are two important conventions hidden in this table.


### Convention 1. The first axis is *time*

For an N‑dimensional array, screamer always treats `axis=0` as the time
axis and every other axis as a parallel independent series. Concretely:

```python
import numpy as np
from screamer import RollingMean

# 1D: 100 samples of a single time series
x_1d = np.random.randn(100)
RollingMean(5)(x_1d).shape          # (100,)

# 2D: 100 samples of 4 parallel series
x_2d = np.random.randn(100, 4)
RollingMean(5)(x_2d).shape          # (100, 4); each column independent

# 3D: 100 samples × 8 instruments × 3 features per instrument
x_3d = np.random.randn(100, 8, 3)
RollingMean(5)(x_3d).shape          # (100, 8, 3); 24 independent streams
```

This means `screamer.RollingMean(5)(x)` and `np.apply_along_axis(...)` agree
about the time axis without you ever specifying it. It is a deliberate
simplification: there is no `axis=` argument because `axis=0` is always
the time axis.

If you want the rolling operation to run along a different axis, transpose
the array yourself (`x.T`) before passing it.

The input shape is preserved, with one exception. `(T,)` and `(T, 1)` are
different inputs and produce different outputs: a 1-D array stays 1-D, and a
2-D column-vector stays 2-D. The exception is a trailing axis of size 1 on an
array of rank 3 or more, which is read as a single field and consumed (rule 1,
detailed in [Time, assets and fields](#time-assets-and-fields)).

```python
RollingMean(5)(np.random.randn(100)).shape       # (100,)
RollingMean(5)(np.random.randn(100, 1)).shape    # (100, 1)
RollingMean(5)(np.random.randn(100, 4, 1)).shape # (100, 4)
```

State is cleanly isolated between streams: `reset()` is called both at
the start of each call and between every column inside it. You can reuse
the same instance for as many calls as you like and they will be
indistinguishable from constructing a fresh instance each time.


### Convention 2. Eager for collections, lazy for iterators

Lists, tuples, and NumPy arrays are processed all at once, in one C++ pass.
Iterators and generators are processed one value at a time: screamer wraps them
in `LazyEvalIterator` and produces each value only when you advance the iteration.
That separation is what makes the live-event use case work, your
generator can yield from a socket, a Kafka stream, a clock-driven simulator,
and screamer applies the algorithm at exactly the same cadence.

```python
from screamer import RollingMean

mean = RollingMean(5)

# Eager: all 100 values computed up-front, returned as an array
result_eager = mean(np.arange(100.0))           # numpy.ndarray, shape (100,)

# Lazy: same algorithm, results come out one at a time
def stream():
    for x in some_live_source():
        yield x

for y in mean(stream()):                        # screamer.LazyEvalIterator
    publish(y)                                  # back-pressure preserved
```

The discriminator is whether the input is a Python `list`/`tuple` /
`numpy.ndarray` (eager paths) or some other iterable (lazy path). A list
or tuple is handled by its own eager branch and returns a Python `list`;
a NumPy array is handled by the array branch and returns an array. The
fact that lists are technically iterable does not matter, the
list/tuple and array branches are checked before the iterable branch in
the dispatcher.

Async iterables are handled the same way through `LazyAsyncIterator`, for every
operator: each `__anext__` awaits one event from every source and runs it
through the same per-event code as `LazyEvalIterator`, so both give the batch
result.

```python
async for y in mean(live_feed()):               # screamer.LazyAsyncIterator
    publish(y)
```

[Streams in depth](streaming.md) covers stream state, several sources, asset
events, and errors.

> **Why no lazy NumPy?** A NumPy array is already in memory, so eager
> processing is strictly faster; lazy iteration over an array would buy
> nothing and pay for `__next__` overhead per element.


## The dispatch order

The exact decision tree implemented in `ScreamerBase::operator()`
(`src/screamer/common/base.cpp`) is:

1. **Is it a scalar?** A scalar is any of:
   - Python `float`, `int`, `bool`,
   - NumPy `float32`, `float64`, `int32`, `int64`, `uint32`, `uint64`.

   The check is `can_cast_to_double(obj)` in
   `include/screamer/common/cast_double.h`. If yes, call
   `process_scalar(value)` and return a Python `float`.

2. **Is it an async iterable?** (Its type defines `__aiter__`.) Wrap in
   `LazyAsyncIterator`. Each `__anext__` awaits one event and produces one
   output.

3. **Is it a tuple whose first item is an array?** Raise `TypeError`. Such a
   tuple is most likely a `(values, index)` stream: pass it to a stream
   operator (`Resample`, `CombineLatest`) or a `Pipeline`, or unpack it and
   pass the values, `op(values)`, keeping the index as it was.
   Row operators (`CrossRank`, `PortfolioReport` and the like) are the
   exception: they read a tuple or list of arrays as rows over time and never
   need the index, so unpack a `(values, index)` pair before calling them.

4. **Is it a list whose first item is an array?** It is a sequence of asset
   events. Each array is one event with one value per asset, and the result
   is a `list` with one array per event.

5. **Is it any other list or tuple?** Process each element eagerly in one
   pass and return a Python `list` of the same length. The container type is
   preserved; a list or tuple is not converted to a NumPy array.

6. **Is it a NumPy array?** A 0‑D array is treated as a scalar and returns a
   `float`. Any array of rank ≥ 1 routes to `process_python_array`, the
   multi‑dimensional handler described in *Convention 1*, and the output
   shape equals the input shape. A length‑1 1‑D array returns a 1‑D array of
   shape `(1,)`, not a scalar.

7. **Is it an iterable?** Wrap in `LazyEvalIterator`. Iteration is lazy; each
   `next()` advances the source and produces one output.

8. **Anything else** → `ValueError("Unsupported input type for call: ...")`.
   (Multi-input functors instead raise `TypeError` for a wrong argument
   count or mixed input kinds, and `ValueError` for shapes that do not
   broadcast; see the multi-input section.)


### Some concrete consequences of this order

- `obj([1, 2, 3])` returns a Python `list` of length 3, not a NumPy array and
  not a lazy iterator, because lists are matched in step 5 before the generic
  iterable branch.
- `obj([1])` returns a `list` of length 1, not a `float`; a length‑1 list does
  not collapse to a scalar.
- `obj(np.array([1.0]))` returns a NumPy array of shape `(1,)`, not a scalar;
  a length‑1 array keeps its rank. Only a 0‑D array collapses to a `float`.
- `obj(iter([1, 2, 3]))` returns a lazy iterator (`LazyEvalIterator`), because
  `iter(...)` is iterable but not a list, tuple, or array.
- `obj(np.float32(2.5))` returns a Python `float`, because NumPy scalars
  are recognised in step 1.
- `obj(decimal.Decimal("1.5"))` raises `ValueError`. `Decimal` is not in the
  scalar table; we deliberately avoid silent precision conversion.
- 8‑bit and 16‑bit NumPy integers are also not in the scalar table. If
  this becomes annoying for a use case, raise an issue, adding them is a
  one‑line change in `cast_double.h`.


## Time, assets and fields

Every array screamer reads has the shape `(T, *A, F?)`:

- `T`, **time**, is always axis 0.
- `A`, **assets**, are zero or more axes of independent series. Each asset has
  its own operator state.
- `F`, **fields**, is an optional last axis holding the parts of one record:
  the inputs of a multi-input operator packed into one array, the outputs of a
  multi-output operator, or a record such as an OHLC bar.

One time step, of shape `(*A, F?)`, is an **event**. A batch array is events
stacked along `T`, and a stream yields the same events one at a time (see
[Streams in depth](streaming.md)). The axes are read as follows:

- **Each asset has its own state.** A one-input operator runs every asset
  column separately, so column `k` of `RollingMean(5)(X)` equals
  `RollingMean(5)(X[:, k])`.
- **Multi-input operators broadcast over assets.** Each input is one field of
  shape `(T, *A_i)`. The asset shapes broadcast with NumPy rules, and `T` must
  match: column `k` of `RollingCorr(20)(X, y)` equals
  `RollingCorr(20)(X[:, k], y)`. A single packed array is split along its last
  axis into the inputs, so a `(T, K, 2)` array feeds `RollingCorr` as `K`
  pairs.
- **A trailing size-1 axis is consumed.** On an array of rank 3 or more, a
  last axis of size 1 is the single input field: `(T, K, 1)` reads as
  `(T, K)`, so `RollingMean(5)` returns `(T, K)` and `BollingerBands(20)`
  returns `(T, K, 3)`. A `(T, 1)` array stays one asset and gives `(T, 1)`.
- **Time operations work per column and append fields.**
  `Resample(freq=60, agg="ohlc")` turns `(T, X)` into `(T', X, 4)`, one bar
  per column.
- **Combining streams joins fields.** `CombineLatest()(bars, signal)` with
  `bars (T, 4)` and `signal (T,)` gives `(T, 5)`.
- **Only row operators read across assets.** `CrossRank()(X)` ranks the `K`
  assets of each row; every other operator keeps assets apart.

```python
X = np.random.randn(100, 4)                     # 4 assets
y = np.random.randn(100)                        # one market series
RollingCorr(20)(X, y).shape                     # (100, 4)
RollingCorr(20)(np.stack([X, 2 * X], axis=-1)).shape  # (100, 4)
RollingMean(5)(X[:, :, None]).shape             # (100, 4)
RollingMean(5)(X[:, :1]).shape                  # (100, 1)
```

A bare `(T, X)` array does not say whether its `X` columns are assets or
fields. The reading matters in two places only:

| Where | Fields reading | Assets reading |
|---|---|---|
| `CombineLatest` of a 2-D and a 1-D stream | concatenate: `(T, X + 1)` | broadcast: `(T, X, 2)` |
| `Pipeline` inputs | one event of `X` fields | `X` assets, `X` operator states (not supported yet) |

Everywhere else both readings give the same numbers: one-input operators treat
columns independently, broadcasting sees only separate arguments, packed
arrays are fields by necessity, multi-output results append fields, row
operators read assets by definition, and `Resample` aggregates per column. The
defaults are that compute operators treat `(T, X)` as independent columns and
stream operators treat `(T, X)` as fields.

A **layout** string states another reading. It names the axes of the data in
order with the letters `T`, `A`, and `F`: `"TA"` says that the columns of a
`(T, K)` array are assets. [`reshape`](functions_streams/reshape.md) applies a
layout to any container, and `CombineLatest(layout=...)` and
`from_pandas(layout=...)` take the same argument. `from_pandas` reads a
DataFrame with complete two-level columns as `(T, A, F)` only with
`layout="TAF"`; without a layout it returns the flat `(T, A*F)` values.


## The multi-input contract (`FunctorBase<_, N, 1>`)

Multi-input classes accept the same conceptual shapes, scalars, arrays,
streams, but in `N` parallel slots. `RollingCorr(window_size)` is the
reference example with `N = 2`.

| You pass... | You get back... |
|---|---|
| `obj(x, y)` (`N` positional scalars) | `float` |
| `obj((x, y))` (one tuple of `N` scalars) | `float` |
| `obj([(x1, y1), (x2, y2), ...])` (list of `N`-tuples) | `list[float]` of the same length |
| `obj(x_arr, y_arr)` - `N` parallel 1D arrays of shape `(T,)` | `numpy.ndarray` of shape `(T,)` |
| `obj(X, Y)` - `N` parallel 2D arrays of shape `(T, K)` | `numpy.ndarray` of shape `(T, K)`, where column `k` is `obj(X[:, k], Y[:, k])` (bit-exact) |
| `obj(X, Y)` - `N` parallel N-D arrays `(T, J, K, ...)` | same shape; `J*K*...` independent paired streams |
| `obj(X_view, Y_view)` - strided views | works; result is contiguous |
| `obj(x_iter, y_iter)` - `N` parallel iterables | a lazy iterator (`LazyEvalIterator`), advancing every input in lock-step and yielding one `float` per step until the first input stops |
| `obj(it)` - one iterable whose events hold all `N` fields (tuples of `N` numbers, or `(K, N)` arrays) | a lazy iterator (`LazyEvalIterator`), as for `N` parallel iterables |
| `obj(x_aiter, y_aiter)` - `N` parallel async iterables, or one async iterable of `N`-field events | an async iterator (`LazyAsyncIterator`) awaiting one event per step; the same results as the iterator forms |
| A mix of async and synchronous inputs | `TypeError` |
| `obj(X, y)` - arrays whose asset shapes broadcast, e.g. `(T, K)` and `(T,)` | `numpy.ndarray` of shape `(T, K)`, where column `k` is `obj(X[:, k], y)`; the `(T,)` input is read once per column, not copied |
| `obj(A)` - a single 2-D array of shape `(T, N)` | `numpy.ndarray` of shape `(T,)` - the `N` columns are the `N` inputs; column `j` → input `j`. Accepted iff `A.shape[1] == N`; any other single-array shape is a `ValueError`. |
| `obj(A)` - a single packed array of shape `(T, K, N)` | `numpy.ndarray` of shape `(T, K)`: `K` assets, each with `N` input fields |
| Asset shapes that do not broadcast, or a different length along time | `ValueError` ("do not broadcast", "same length along time") |
| Mixed kinds (one scalar + one array, etc.) or a wrong number of inputs | `TypeError` |

This is the array form of `obj(A[:, 0], A[:, 1], ...)`, and is what makes
`RollingCorr(w)(CombineLatest()(a, b)[0])` work directly.

```python
from screamer import RollingCorr

corr = RollingCorr(window_size=20)

# Scalar pair (returns a float)
corr(1.5, 2.0)

# Two parallel arrays (returns an array)
returns_a = np.diff(np.log(price_a))
returns_b = np.diff(np.log(price_b))
corr(returns_a, returns_b)

# Streaming list of pairs (returns a list)
ticks = [(1.5, 2.0), (1.6, 2.1), (1.4, 1.9), ...]
corr(ticks)

# Two parallel generators
for c in corr(gen_a(), gen_b()):
    publish(c)
```

### Multi-D semantics for paired arrays

The 2D / N-D rules are the obvious generalisation of the single-input
convention: **axis 0 is time across both inputs, every higher axis is
treated as independent paired streams.** Concretely, for two `(T, K)`
arrays:

```python
corr = RollingCorr(window_size=10)
X = np.random.randn(100, 4)   # 4 series of x
Y = np.random.randn(100, 4)   # 4 series of y, paired with X column-wise

result = corr(X, Y)           # shape (100, 4)
# result[:, k] == corr(X[:, k], Y[:, k])  for every k    (bit-exact)
```

The pairing is **column-by-column**: `X[:, k]` is correlated against
`Y[:, k]` only. Cross-pair correlations (`X[:, j]` vs `Y[:, k]` for
`j != k`) are not computed, that would be a different operator
returning a `(T, K, K)` result and is out of scope for `RollingCorr`.

The streams are independent: `reset()` is called between columns inside
a single batch call, just like for the single-input array path.

### Caveats specific to multi-input

- The eager/lazy rule is the same as for single input. `N` parallel
  concrete collections (lists, tuples, arrays) are processed eagerly in
  one pass; `N` parallel lazy iterables (generators, `iter(...)`) return
  a lazy iterator that advances every input in lock-step and yields one
  result per step, until the first input stops. Feed generators
  directly:

  ```python
  corr = RollingCorr(window_size=20)
  for c in corr(gen_a(), gen_b()):
      publish(c)
  ```

- All `N` inputs must be the same kind (all scalars, all numpy arrays,
  all iterables). Mixing kinds raises `TypeError`. Their shapes may differ
  where they broadcast: a `(T,)` market series pairs with every column of a
  `(T, K)` panel.


## The multi-output contract (`FunctorBase<_, 1, M>`)

Some functions produce more than one value per time step. `RollingMinMax`
returns the pair `(min, max)`; `BollingerBands` returns the triple
`(lower, mid, upper)`. The call shape mirrors the single-input one with
one rule added: the output gets an extra trailing axis of size `M`.

| You pass... | You get back... |
|---|---|
| scalar | Python `tuple` of `M` floats |
| 1-D array of shape `(T,)` | NumPy array of shape `(T, M)` |
| 2-D array of shape `(T, K)` | NumPy array of shape `(T, K, M)` |
| N-D array of shape `(T, ..., K)` | NumPy array of shape `(T, ..., K, M)` |
| iterable | a lazy iterator (`LazyEvalIterator`) yielding one `tuple` of `M` floats per step |
| async iterable | an async iterator (`LazyAsyncIterator`) awaiting one `tuple` of `M` floats per step |

The shape rule is `output.shape == input.shape + (M,)`, after a trailing
size-1 axis on an input of rank 3 or more has been consumed. `(T,)` and
`(T, 1)` produce different outputs, `(T, M)` and `(T, 1, M)`, while
`(T, K, 1)` gives `(T, K, M)`.

```python
from screamer import RollingMinMax, BollingerBands

RollingMinMax(5)(np.random.randn(100)).shape         # (100, 2)
RollingMinMax(5)(np.random.randn(100, 4)).shape      # (100, 4, 2)
BollingerBands(20)(np.random.randn(100)).shape       # (100, 3)
BollingerBands(20)(np.random.randn(100, 4)).shape    # (100, 4, 3)
```

Per-step access uses the trailing axis: `bb[:, 0]` is the lower band,
`bb[:, 1]` the mid, `bb[:, 2]` the upper. For a 2-D input, it would be
`bb[:, k, 0]`, `bb[:, k, 1]`, `bb[:, k, 2]` for each parallel series `k`.

Like the multi-input path, the iterable case is lazy: a generator or
`iter(...)` returns a lazy iterator yielding one `tuple` of `M` floats
per step, while a list, tuple, or array is processed eagerly.


## The multi-input multi-output contract (`FunctorBase<_, N, M>`)

Functions that map `N` parallel input streams to `M` parallel output streams compose the two rules above: inputs are paired column-by-column (from the `N → 1` path) and outputs gain a trailing axis of size `M` (from the `1 → M` path). `Cart2Polar` and `Polar2Cart` are reference examples with `N = M = 2`.

| You pass... | You get back... |
|---|---|
| `obj(x, y)` (`N` positional scalars) | tuple of `M` floats |
| `obj((x, y))` (one tuple of `N` scalars) | tuple of `M` floats |
| `obj([(x1, y1), (x2, y2), ...])` (list of `N`-tuples) | `list[tuple[float, ...]]` of the same length |
| `N` parallel 1D arrays of shape `(T,)` | NumPy array of shape `(T, M)` |
| `N` parallel 2D arrays of shape `(T, K)` | NumPy array of shape `(T, K, M)`; column `k` is `obj(X[:, k], Y[:, k])` (bit-exact) |
| `N` parallel iterables, or one iterable of `N`-field events | a lazy iterator (`LazyEvalIterator`) yielding one `tuple` of `M` floats per step |
| `N` parallel async iterables, or one async iterable of `N`-field events | an async iterator (`LazyAsyncIterator`) awaiting one `tuple` of `M` floats per step |
| `obj(A)` - a single 2-D array of shape `(T, N)` | NumPy array of shape `(T, M)` - the `N` columns are the `N` inputs; column `j` → input `j`. Accepted iff `A.shape[1] == N`; any other single-array shape is a `ValueError`. |

The shape rule is `output.shape == broadcast_input.shape + (M,)`, where the input asset shapes broadcast as in the `N → 1` case and shapes that do not broadcast raise `ValueError`. The dispatcher calls `reset()` between independent paired streams in a 2D/N-D batch, so stateful `N → M` functors don't leak state across columns.

```python
from screamer import Cart2Polar, Polar2Cart

# Two scalars -> tuple of two floats
Cart2Polar()(3.0, 4.0)               # (5.0, 0.9272...)

# Two 1D arrays -> shape (T, 2)
Cart2Polar()(np.random.randn(100), np.random.randn(100)).shape    # (100, 2)

# Two 2D arrays -> shape (T, K, 2), paired column-by-column
Cart2Polar()(np.random.randn(100, 4), np.random.randn(100, 4)).shape  # (100, 4, 2)

# Roundtrip: Polar2Cart and Cart2Polar are inverses
polar = Cart2Polar()(x, y)
back = Polar2Cart()(polar[:, 0], polar[:, 1])    # equals (x, y)
```


## Symmetry table

The same algorithm produces the same numbers across every input shape
(modulo `NaN`/precision noise around warmup):

| Property | Holds? |
|---|---|
| `obj(x_array)[i] == obj(scalar_stream)[i]` | ✓ exact (modulo IEEE float reorderings during reductions, which are not done) |
| `obj(2d_array)[:, k] == obj(1d_array_k)` | ✓ exact |
| `obj(strided_view) == obj(strided_view.copy())` | ✓ exact |
| Same instance, called twice on the same data | ✓ identical output (regression-tested in `tests/test_nan_warmup.py`) |
| Reusing an instance vs. constructing a fresh one | ✓ identical (one `reset()` happens automatically before and after each batch call) |


## State, reset, and causality

- All state lives on the instance. `reset()` zeroes it out.
- The eager array paths call `reset()` at the start of every call and at
  the end. So passing a NumPy array is functionally equivalent to
  constructing a fresh instance for that batch.
- Between independent series in a 2D/3D array, `reset()` is called
  automatically, column `k` does not see column `k‑1`'s state.
- A call with one number, `op(x)`, is one step of the operator's running
  series and keeps its state from call to call. `reset()` starts the running
  series over.
- A stream (iterator or async iterator) is an independent run: it resets the
  running series when it opens, then runs on copies of the operator that it
  owns, so it never shares state with live calls or with another stream.
  [Streams in depth](streaming.md) has the details.
- **Causal.** Output at index `t` depends only on inputs at indices `<= t`;
  no function looks ahead.

[NaN and warmup](nan_and_warmup.md) documents warmup, the leading region where a
function has not yet seen enough samples, and the `start_policy` argument
(`"strict"`, `"expanding"`, `"zero"`) that controls it.


## Design notes

The single polymorphic dispatch is the load-bearing element of the API, for
four reasons:

1. **One mental model for two runtimes.** A backtest reads from a pandas
   DataFrame; a live system reads from a queue, a websocket, or a clock-driven
   simulator. Both route to the same C++ inner loop, because the dispatcher is
   the only place that touches the input surface.

2. **Numbers are bit-exact across shapes.** There is no training/serving skew:
   if live values disagree with a backtest, the cause is a difference in the
   data feed, not the library.

3. **Adding an algorithm costs nothing on the dispatch side.** A new class
   implements `process_scalar(double)` (and optionally the array fast paths),
   and inherits all input shapes.

4. **The multi-dimensional convention is explicit.** There is no `axis=`
   argument to forget: time is always axis 0, everything else is parallel.


## See also

- `tests/test_io_size.py`, exhaustively walks every shape for every
  algorithm and asserts the output shape and dtype are correct.
- `tests/test_view.py`, strided‑view correctness across the full library.
- `tests/test_stream_vs_batch.py`, proves the eager (array) path and the
  scalar path produce the same numbers.
- `tests/test_stream_vs_generator.py`, proves the eager array path and
  the lazy generator path produce the same numbers.
- `tests/test_rolling_corr.py`, the multi-input contract for `N=2`.
