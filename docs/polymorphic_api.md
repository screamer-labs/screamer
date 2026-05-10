# Polymorphic input/output behavior

A defining feature of screamer is that **the same callable works for every
input shape**, a single scalar, a NumPy array, a strided view, a list, an
iterator, an async generator. The same code you write to backtest on a
historical dataset runs unchanged in a live event loop. This page is the
contract: exactly what each input type does and what each returns.

The contract has two layers:

- **1‑input / 1‑output classes** (`RollingMean`, `EwVar`, `Diff`, `Lag`,
  every `Rolling*`, every `Ew*`, all the math transforms, ...). These inherit
  from `ScreamerBase`. The vast majority of screamer.
- **N‑input / 1‑output classes** (`RollingCorr`, `RollingCov`,
  `RollingBeta`, `RollingSpread`). These inherit from
  `FunctorBase<Derived, N, 1>`.
- **1‑input / M‑output classes** (`RollingMinMax`, `BollingerBands`).
  These inherit from `FunctorBase<Derived, 1, M>`.

The general N‑input / M‑output case is not yet implemented; calling such
a class raises `TypeError: Unsupported functor type`.


## The single-input contract (`ScreamerBase`)

Every 1‑in/1‑out class supports the following input/output shapes:

| You pass... | You get back... | How |
|---|---|---|
| `int`, `float`, `bool` | `float` | one `process_scalar` call |
| `numpy.float32/64`, `numpy.int32/64`, `numpy.uint32/64` | `float` | one `process_scalar` call |
| 1D NumPy array, length 1 | `float` | unwrapped, treated as scalar |
| 1D NumPy array, length ≥ 2 | 1D NumPy array, same shape | one `process_*` pass over the buffer |
| 1D strided NumPy view | 1D NumPy array, same shape | strided pass; output is contiguous |
| 2D NumPy array `(T, K)` | 2D NumPy array `(T, K)` | one independent stream per column; `reset()` between columns |
| N‑D NumPy array `(T, J, K, ...)` | same shape | `J·K·...` independent streams along axis 0 |
| Python `list` of numbers | NumPy array of same length | converted to 1D `array_t<double>`, processed in one pass |
| Python `tuple` of numbers | NumPy array of same length | same |
| Python iterator (`iter(...)`, generator, anything iterable that is not list/tuple/array) | a screamer `LazyIterator` | results yielded **one at a time on demand** |
| Async generator (`async def` with `yield`) | a screamer `LazyAsyncIterator` | results awaited one at a time |
| Anything else | `TypeError` | "Unsupported input type" |

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

The input shape is preserved exactly. `(T,)` and `(T, 1)` are different
inputs and produce different outputs: a 1-D array stays 1-D, a 2-D
column-vector stays 2-D. screamer never silently squeezes or expands
axes.

```python
RollingMean(5)(np.random.randn(100)).shape       # (100,)
RollingMean(5)(np.random.randn(100, 1)).shape    # (100, 1)
```

State is cleanly isolated between streams: `reset()` is called both at
the start of each call and between every column inside it. You can reuse
the same instance for as many calls as you like and they will be
indistinguishable from constructing a fresh instance each time.


### Convention 2. Eager for collections, lazy for iterators

Lists, tuples, and NumPy arrays are processed **eagerly** in one C++ pass.
Iterators and generators are processed **lazily**: screamer wraps them in
`LazyIterator` and produces values only when you advance the iteration.
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

for y in mean(stream()):                        # screamer.LazyIterator
    publish(y)                                  # back-pressure preserved
```

The discriminator is whether the input is a Python `list`/`tuple` /
`numpy.ndarray` (eager paths) or some other iterable (lazy path). A list
is *first* materialised into a NumPy array and then processed in one shot;
the fact that lists are technically iterable does not matter, the
list/array branch is checked before the iterable branch in the dispatcher.

Async generators are handled symmetrically through `LazyAsyncIterator`.

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

2. **Is it a NumPy array, list, or tuple?** Cast to
   `numpy.ndarray<float64>`. Two sub‑cases:
   - **Length 1**: extract the single element and treat it as a scalar.
   - **Length ≥ 2**: route to `process_python_array`, which is the
     multi‑dimensional handler described in *Convention 1*.

3. **Is it an iterable?** Wrap in `LazyIterator`. Iteration is lazy; each
   `next()` advances the source and produces one output.

4. **Is it an async generator?** (`hasattr(obj, "__aiter__")` and
   `hasattr(obj, "__anext__")`.) Wrap in `LazyAsyncIterator`.

5. **Anything else** → `TypeError("Unsupported input type for call: ...")`.


### Some concrete consequences of this order

- `obj([1, 2, 3])` returns a NumPy array of length 3, not a `LazyIterator`,
  because lists are matched in step 2 before reaching the iterable branch.
- `obj([1])` returns a `float`, not a length‑1 array.
- `obj(iter([1, 2, 3]))` returns a `LazyIterator`, because `iter(...)` is
  iterable but not a list, tuple, or array.
- `obj(np.float32(2.5))` returns a Python `float`, because NumPy scalars
  are recognised in step 1.
- `obj(decimal.Decimal("1.5"))` raises `TypeError`. `Decimal` is not in the
  scalar table; we deliberately avoid silent precision conversion.
- 8‑bit and 16‑bit NumPy integers are also not in the scalar table. If
  this becomes annoying for a use case, raise an issue, adding them is a
  one‑line change in `cast_double.h`.


## The multi-input contract (`FunctorBase<_, N, 1>`)

Multi-input classes accept the same conceptual shapes, scalars, arrays,
streams, but in `N` parallel slots. `RollingCorr(window_size)` is the
reference example with `N = 2`.

| You pass... | You get back... |
|---|---|
| `obj(x, y)` (`N` positional scalars) | `float` |
| `obj((x, y))` (one tuple of `N` scalars) | `float` |
| `obj([(x1, y1), (x2, y2), ...])` (list of `N`-tuples) | `list[float]` of the same length |
| `obj(x_arr, y_arr)` -- `N` parallel 1D arrays of shape `(T,)` | `numpy.ndarray` of shape `(T,)` |
| `obj(X, Y)` -- `N` parallel 2D arrays of shape `(T, K)` | `numpy.ndarray` of shape `(T, K)`, where column `k` is `obj(X[:, k], Y[:, k])` (bit-exact) |
| `obj(X, Y)` -- `N` parallel N-D arrays `(T, J, K, ...)` | same shape; `J*K*...` independent paired streams |
| `obj(X_view, Y_view)` -- strided views | works; result is contiguous |
| `obj(x_iter, y_iter)` -- `N` parallel iterables | `list[float]` (advanced in lock-step until the first stops) |
| Mismatched shapes or ndim across the `N` inputs | `TypeError` with a clear message |
| Mixed kinds (one scalar + one array, etc.) | `TypeError` |

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

- The `N` parallel iterables case is **eager**, not lazy: the helper
  builds a `std::vector<double>` of all results before returning. There
  is no `LazyIterator` at the moment for `N > 1`. If you want truly
  streaming multi-input processing today, feed the values yourself in
  a loop:

  ```python
  corr = RollingCorr(window_size=20)
  for x, y in zip(stream_a, stream_b):
      yield corr(x, y)
  ```

  (Each scalar call is constant time and matches the dispatcher's first
  row in the table.)

- All `N` inputs must be the same kind (all scalars, all numpy arrays,
  all iterables) and the same shape. Mixing them raises `TypeError`.


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
| iterable | `list[tuple[float, ...]]` of length matching the input (eager) |

The shape rule is exactly: `output.shape == input.shape + (M,)`. The same
input-axis-preservation guarantee from the single-output path holds here
too. `(T,)` and `(T, 1)` produce different outputs: `(T, M)` and `(T, 1, M)`
respectively.

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

Like the multi-input path, the iterable case is eager: it returns
`list[tuple[...]]`, not a lazy iterator.


## The multi-input multi-output contract (`FunctorBase<_, N, M>`)

Functions that map `N` parallel input streams to `M` parallel output streams compose the two rules above: inputs are paired column-by-column (from the `N → 1` path) and outputs gain a trailing axis of size `M` (from the `1 → M` path). `Cart2Polar` and `Polar2Cart` are reference examples with `N = M = 2`.

| You pass... | You get back... |
|---|---|
| `obj(x, y)` (`N` positional scalars) | tuple of `M` floats |
| `obj((x, y))` (one tuple of `N` scalars) | tuple of `M` floats |
| `obj([(x1, y1), (x2, y2), ...])` (list of `N`-tuples) | `list[tuple[float, ...]]` of the same length |
| `N` parallel 1D arrays of shape `(T,)` | NumPy array of shape `(T, M)` |
| `N` parallel 2D arrays of shape `(T, K)` | NumPy array of shape `(T, K, M)`; column `k` is `obj(X[:, k], Y[:, k])` (bit-exact) |
| `N` parallel iterables | `list[tuple[float, ...]]` (eager) |

The shape rule is exactly: `output.shape == single_input.shape + (M,)`. Mismatched shapes across the `N` inputs raise `TypeError`, same as `N → 1`. The dispatcher calls `reset()` between independent paired streams in a 2D/N-D batch, so stateful `N → M` functors don't leak state across columns.

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
(modulo `NaN`/precision noise around warmup). That property is what makes
the dual training/live workflow practical:

| Property | Holds? |
|---|---|
| `obj(x_array)[i] == obj(scalar_stream)[i]` | ✓ exact (modulo IEEE float reorderings during reductions, which are not done) |
| `obj(2d_array)[:, k] == obj(1d_array_k)` | ✓ exact |
| `obj(strided_view) == obj(strided_view.copy())` | ✓ exact |
| Same instance, called twice on the same data | ✓ identical output (regression-tested in `tests/test_nan_warmup.py`) |
| Reusing an instance vs. constructing a fresh one | ✓ identical (one `reset()` happens automatically before and after each batch call) |


## State, reset, and warmup

- All state lives on the instance. `reset()` zeroes it out.
- The eager array paths call `reset()` at the start of every call and at
  the end. So passing a NumPy array is functionally equivalent to
  constructing a fresh instance for that batch.
- Between independent series in a 2D/3D array, `reset()` is called
  automatically, column `k` does not see column `k‑1`'s state.
- The lazy iterator path does **not** call `reset()` for you. The whole
  point of streaming is preserving state across `__next__` calls. If you
  want to start over, construct a new instance or call `instance.reset()`
  yourself.
- Warmup semantics depend on the algorithm and on its `start_policy`
  argument (`"strict"`, `"expanding"`, `"zero"`). Strict policies emit
  `NaN` until enough samples have arrived; expanding policies compute
  with whatever is available. The default everywhere is `"strict"`.


## Why this design

The polymorphism is not an accident or a convenience layer, it is the
load‑bearing element of the API.

1. **One mental model for two very different runtimes.** A backtest reads
   from a Pandas DataFrame; a live system reads from a queue, a websocket,
   or a clock-driven simulator. Both shapes route to the same C++ inner
   loop because the dispatcher is the only place that touches the input
   surface.

2. **The numbers are bit-exact across shapes.** No "training/serving
   skew." The cause of "live values disagree with what we backtested" is
   reduced to: did your live data feed match your historical data? The
   library cannot be the source of disagreement.

3. **Adding a new algorithm costs nothing on the dispatch side.** A new
   class only has to implement `process_scalar(double)` (and optionally
   the array fast paths). All ten input shapes are inherited.

4. **The multi-dimensional convention is explicit.** No `axis=` argument
   to forget. No accidental "you transposed your data and got nonsense."
   Time is always axis 0. Anything else is parallel.


## See also

- `tests/test_io_size.py`, exhaustively walks every shape for every
  algorithm and asserts the output shape and dtype are correct.
- `tests/test_view.py`, strided‑view correctness across the full library.
- `tests/test_stream_vs_batch.py`, proves the eager (array) path and the
  scalar path produce the same numbers.
- `tests/test_stream_vs_generator.py`, proves the eager array path and
  the lazy generator path produce the same numbers.
- `tests/test_rolling_corr.py`, the multi-input contract for `N=2`.
