# Using `screamer` Functions

Every `screamer` function follows the same two-step pattern:

1. Create an object with the parameters you want, for example
   `obj = RollingMean(window_size=30)`.
2. Call the object on your data, for example `result = obj(data)`.

The object holds the algorithm state. The call applies it. You can call
the same object many times.

The same object accepts a single number, a NumPy array, a Python list,
or a generator. The output type matches the input type. This is what
makes the same code work for backtests on historical arrays and for live
event loops on streams.

---

## One scalar at a time

The simplest use is feeding values one by one.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   np.random.seed(42)
   # --- hide: stop ---
   rolling = RollingMean(window_size=3)

   for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
       y = rolling(x)
       print(f"in: {x:.1f}   out: {y:.4f}")
```

The first two outputs are `nan` because the window of size 3 is not full
yet. From the third value onwards the rolling mean is defined.

---

## A whole array at once

Pass a NumPy array and get a NumPy array of the same shape back.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   np.random.seed(42)
   # --- hide: stop ---
   data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

   result = RollingMean(window_size=3)(data)
   print(result)
```

This produces the same numbers as the scalar loop above. Calling the
function on an array is equivalent to looping over the array and feeding
each value to a fresh object. Both forms are supported so you can pick
whichever is convenient.

You can also call the constructor and apply the function in a single
expression:

```python
result = RollingMean(window_size=3)(data)
```

or build the object first and reuse it across several arrays:

```python
rolling = RollingMean(window_size=3)
result_a = rolling(data_a)   # fresh state, then reset
result_b = rolling(data_b)   # fresh state again
```

When you pass an array, the object resets its internal state at the
start of the call and at the end. Two array calls on the same object
behave the same as two array calls on two fresh objects.

---

## Time runs along axis 0

For 2-D arrays, `screamer` reads the **first axis as time** and every
other axis as parallel independent series. A `(100, 4)` array means
100 time samples for 4 series.

![Column processing](img/per_column.png "Processing columns separately as
individual streams")

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   np.random.seed(0)
   # --- hide: stop ---
   data = np.array([
       [1.0,  10.0],   # t=0
       [2.0,  20.0],   # t=1
       [3.0,  30.0],   # t=2
       [4.0,  40.0],   # t=3
       [5.0,  50.0],   # t=4
   ])

   result = RollingMean(window_size=3)(data)
   print(result)
```

Column 0 and column 1 are processed independently. The state from one
column does not leak into the next, because the object resets between
columns.

This means `RollingMean(3)(data)[:, k]` is equivalent to
`RollingMean(3)(data[:, k])` for every column `k`. The same convention
applies to higher dimensions: a `(T, J, K)` array is treated as `J * K`
parallel streams of length `T`.

If you want the rolling operation to run along a different axis,
transpose the array yourself before passing it.

---

## Iterators and generators (streaming)

When you pass a Python iterator or generator, the function returns a
lazy iterator that produces values one at a time as you advance it. The
state is preserved across calls to `next()`, which is exactly what
live event loops need.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMean
   np.random.seed(42)
   # --- hide: stop ---
   def live_feed():
       for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
           yield x

   rolling = RollingMean(window_size=3)
   for y in rolling(live_feed()):
       print(f"{y:.4f}")
```

The output matches the array form and the scalar loop. The same
algorithm, the same numbers, three different input shapes.

> Lists and tuples take the eager path: `obj([1, 2, 3])` returns a
> NumPy array of length 3. Use `iter([1, 2, 3])` if you want the lazy
> path.

---

## Composing functions

Because every function accepts the output of every other function,
chaining is straightforward. Here `Diff` produces 3-step differences,
and `RollingMax` takes the rolling maximum of those differences.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingMax, Diff
   np.random.seed(42)
   # --- hide: stop ---
   data = np.random.normal(size=10)

   diff = Diff(3)
   rmax = RollingMax(window_size=4)
   result = rmax(diff(data))

   print(result)
```

The same chain works on a generator without any change:

```python
chained = rmax(diff(live_feed()))
for y in chained:
    publish(y)
```

---

## Two-input functions

Some functions take more than one stream. `RollingCorr` is the rolling
Pearson correlation of two series. The call shape mirrors the
single-input one: scalars produce a scalar, arrays produce an array,
generators produce a lazy iterator.

```{eval-rst}
.. exec_code::

   # --- hide: start ---
   import numpy as np
   from screamer import RollingCorr
   np.random.seed(42)
   # --- hide: stop ---
   x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
   y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

   result = RollingCorr(window_size=3)(x, y)
   print(result)
```

`y` is `2 * x`, so the correlation is `1.0` once the window is full.

For 2-D arrays, the column-by-column rule still applies. Two arrays of
shape `(T, K)` produce a `(T, K)` result, where column `k` is the
rolling correlation of `X[:, k]` against `Y[:, k]`. Cross pairs (column
`j` of one with column `k` of the other) are not computed.

```python
X = returns_per_asset                # shape (T, K)
Y = factor_returns                   # shape (T, K)
RollingCorr(window_size=20)(X, Y)    # shape (T, K)
```

For element-by-element streaming, call the object with two scalars:

```python
corr = RollingCorr(window_size=20)
for x, y in zip(stream_a, stream_b):
    c = corr(x, y)
    publish(c)
```

---

## Equivalence in one place

For any `screamer` function `f`, the following all produce the same
sequence of numbers (modulo `nan` placement during warmup):

```python
# 1. Scalar loop
out = []
g = f()
for x in xs:
    out.append(g(x))

# 2. Eager array
out = f()(np.asarray(xs))

# 3. Lazy generator
out = list(f()(iter(xs)))
```

This equivalence is what makes the same code suitable for both batch
analysis and live deployment.

---

## Reset

The object's internal state is reset for you in two situations:

- At the start and end of every batch call (an array, a list, a tuple).
- Between columns of a multi-dimensional array.

The state is **not** reset on the streaming paths (scalar loop, generator,
async generator). If you want a fresh start during streaming, call
`obj.reset()` explicitly or build a new object.

```python
rolling = RollingMean(window_size=3)
rolling(1.0)
rolling(2.0)
rolling.reset()        # back to warmup
```
