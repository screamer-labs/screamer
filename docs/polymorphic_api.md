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
| Python iterator (`iter(...)`, generator, anything iterable that is not list/tuple/array) | a screamer `LazyEvalIterator` | results yielded **one at a time on demand** |
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

2. **Is it a list or tuple?** Process each element eagerly in one pass and
   return a Python `list` of the same length. The container type is
   preserved; a list or tuple is not converted to a NumPy array.

3. **Is it a NumPy array?** A 0‑D array is treated as a scalar and returns a
   `float`. Any array of rank ≥ 1 routes to `process_python_array`, the
   multi‑dimensional handler described in *Convention 1*, and the output
   shape equals the input shape. A length‑1 1‑D array returns a 1‑D array of
   shape `(1,)`, not a scalar.

4. **Is it an iterable?** Wrap in `LazyEvalIterator`. Iteration is lazy; each
   `next()` advances the source and produces one output.

5. **Is it an async generator?** (`hasattr(obj, "__aiter__")` and
   `hasattr(obj, "__anext__")`.) Wrap in `LazyAsyncIterator`.

6. **Anything else** → `TypeError("Unsupported input type for call: ...")`.


### Some concrete consequences of this order

- `obj([1, 2, 3])` returns a Python `list` of length 3, not a NumPy array and
  not a lazy iterator, because lists are matched in step 2 before the generic
  iterable branch.
- `obj([1])` returns a `list` of length 1, not a `float`; a length‑1 list does
  not collapse to a scalar.
- `obj(np.array([1.0]))` returns a NumPy array of shape `(1,)`, not a scalar;
  a length‑1 array keeps its rank. Only a 0‑D array collapses to a `float`.
- `obj(iter([1, 2, 3]))` returns a lazy iterator (`LazyEvalIterator`), because
  `iter(...)` is iterable but not a list, tuple, or array.
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
| `obj(x_arr, y_arr)` - `N` parallel 1D arrays of shape `(T,)` | `numpy.ndarray` of shape `(T,)` |
| `obj(X, Y)` - `N` parallel 2D arrays of shape `(T, K)` | `numpy.ndarray` of shape `(T, K)`, where column `k` is `obj(X[:, k], Y[:, k])` (bit-exact) |
| `obj(X, Y)` - `N` parallel N-D arrays `(T, J, K, ...)` | same shape; `J*K*...` independent paired streams |
| `obj(X_view, Y_view)` - strided views | works; result is contiguous |
| `obj(x_iter, y_iter)` - `N` parallel iterables | a lazy iterator (`LazyEvalIterator`), advancing every input in lock-step and yielding one `float` per step until the first input stops |
| `obj(A)` - a single 2-D array of shape `(T, N)` | `numpy.ndarray` of shape `(T,)` - the `N` columns are the `N` inputs; column `j` → input `j`. Accepted iff `A.shape[1] == N`; any other single-array shape is a `TypeError`/`ValueError`. |
| Mismatched shapes or ndim across the `N` inputs | `TypeError` with a clear message |
| Mixed kinds (one scalar + one array, etc.) | `TypeError` |

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
| iterable | a lazy iterator (`LazyEvalIterator`) yielding one `tuple` of `M` floats per step |

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
| `N` parallel iterables | a lazy iterator (`LazyEvalIterator`) yielding one `tuple` of `M` floats per step |
| `obj(A)` - a single 2-D array of shape `(T, N)` | NumPy array of shape `(T, M)` - the `N` columns are the `N` inputs; column `j` → input `j`. Accepted iff `A.shape[1] == N`; any other single-array shape is a `TypeError`/`ValueError`. |

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
- The lazy iterator path does **not** call `reset()` for you. The whole
  point of streaming is preserving state across `__next__` calls. If you
  want to start over, construct a new instance or call `instance.reset()`
  yourself.
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
