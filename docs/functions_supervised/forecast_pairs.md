---
name: forecast_pairs
title: forecast_pairs
kind: function
short: Build a forecasting training set where row t holds the features at t and the target over the horizon after t.
tags:
- supervised learning
- training set
- features
- labels
- target
- forecasting
see_also:
- walk_forward
- TripleBarrier
- Lag
- Return
topics:
- supervised
covers: []
---

# `forecast_pairs`

`forecast_pairs(X, y, count=h)` is a helper that builds a training set: row `t`
holds the features `X[t]` and the target over the next `h` events. With
`duration=` the horizon is a span of the index instead of a number of events.
The target `y` is a causal series, so its value at step `t` is known at `t`: a
target over a horizon is stamped at the end of that horizon. For example
`LogReturn(h)(price)` at step `t` is the return over `(t - h, t]`, and
`forecast_pairs` moves it back to the row of the features it follows.

A training row and a live row differ only in the target:

| | features | target |
|---|---|---|
| training, row t | `X[t]` | outcome of `(t, t + h]` |
| live, row t | `X[t]` | NaN, not known yet |

The feature matrix is built by the same code, at the same timestamps, in
research and live, and the model learns "features now, outcome next".

`index` chooses whose timestamp a row carries:

| | row holds | NaN padding |
|---|---|---|
| `index="features"` (default) | `X[t]` and the target over `(t, t + h]` | the last `h` targets |
| `index="target"` | `X[t - h]` and `y[t]` | the first `h` feature rows |

In count mode both settings hold the same pairs and differ only in the stamp
and in where the padding is. With `dropna=True` they return identical arrays.

<!-- HELP_END -->

## Parameters

- `X`: features, `(T,)` or `(T, K)`, in count mode; a `(values, index)` tuple
  in duration mode, whose values may also be 1-D or 2-D.
- `y`: target, one column (`(T,)` or `(T, 1)`), in count mode; a
  `(values, index)` tuple in duration mode. Must be causal.
- `count=` (int): the horizon as a number of events. With
  `index="features"`, row `t` holds `y[t + count]`. `X` and `y` tick at the
  same indexes: in a lazy stream or a `Pipeline`, an index that only one of
  them has raises `ValueError`; use `duration=` for inputs on different clocks.
- `duration=` (numeric): the horizon in index units, as in `Delay`. `X` and
  `y` carry an index and may tick on different clocks.
- `index="features"`: `"features"` or `"target"`, as in the table above.
- `dropna=False`: if `True`, drop every row that holds a NaN, such as the rows
  whose target is not known yet.

Exactly one of `count=` and `duration=` is given; both or neither raises
`ValueError`. `count` is from 1 to 2**31 - 1 and `duration` is at least 0; a
value out of range raises `ValueError`. Both are numbers: a string raises
`TypeError`, and a float is truncated, as `Delay` truncates its duration.
`index` is one of the two strings, and another value raises `ValueError`
naming both. `dropna` is a bool.

## Duration mode

Rows follow the clock of the side whose timestamp they carry:

- `index="features"`: one row per `X` event at `t`, holding the last `y` at or
  before `t + duration`. When `y` ends before `t + duration`, the target is
  NaN.
- `index="target"`: one row per `y` event at `t`, holding the last `X` at or
  before `t - duration`, column by column: a NaN in a column of that row gives
  the column's last non-NaN value.

## Returns

A two-tuple `(X_paired, y_paired)` of numpy arrays, in time order. `X_paired`
has the rank of `X`: a 2-D `X` with `K` columns gives a `(rows, K)` array. To
map a row back to its time, keep your own index next to `X` and `y`, such as a
pandas index or the index you pass in duration mode.

## Regimes

The function dispatches on the input type:

- **Eager** (arrays, or `(values, index)` tuples in duration mode): returns
  `(X_paired, y_paired)`.
- **Graph** (`Input` nodes): returns a Node whose output row is `[X..., y]`,
  `K + 1` columns. Wrap it in a `Pipeline` to run it in batch or as a live
  session.
- **Lazy** (generators of `(value, index)` events): returns a lazy iterator of
  `((x..., y), index)` events.

Each regime runs one graph defined in C++, the same one JavaScript's
`forecastPairs` runs. In a stream with `index="features"`, the row for `t` is
emitted once its target is known, stamped `t`, so stamps stay in increasing
order: in count mode after `count` more events, in duration mode once a `y`
event later than `t + duration` arrives. With sparse `y`, a live row can wait
longer than `duration`. At the end of input the rows still waiting are
emitted, with a NaN target where none is known, which gives the eager result.
Memory is the rows waiting for their target, not the dataset.

A `Pipeline` saved before `index` existed loads with `index="target"`.

## JavaScript

`forecastPairs(X, y, { count | duration, index, dropna })` builds the same
graph and returns `[X_paired, y_paired]`, with the same values as Python. In
count mode `X` is a 1-D array, a `[T, K]` NdArray or T rows, and `y` one column
of the same length. In duration mode `X` and `y` are `{ values, index }`
streams. Given `Input` nodes, it returns a Node for a `Pipeline`.

## Examples

The [labels guide](../labels.md) shows which target to pair for each kind of
prediction.

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import LogReturn
    from screamer.supervised import forecast_pairs

    rng = np.random.default_rng(42)
    price = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(8)))
    X = np.column_stack([LogReturn(1)(price), LogReturn(3)(price)])

    h = 2
    y = LogReturn(h)(price)                 # return over (t - h, t]

    Xp, yp = forecast_pairs(X, y, count=h)  # row t: X[t] and the return over (t, t + h]
    print(np.column_stack([Xp, yp]).round(4))
```
