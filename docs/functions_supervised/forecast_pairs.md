---
name: forecast_pairs
title: forecast_pairs
kind: function
short: Build a forecasting training set by aligning lagged features with a future causal target.
tags:
- supervised learning
- training set
- features
- labels
- target
- forecasting
see_also:
- walk_forward
- Lag
- Return
topics:
- supervised
covers: []
---

# `forecast_pairs`

`forecast_pairs` pairs historical features with a future target so that a model
can learn to predict `count` events (or `duration` index-units) ahead. Row `t`
in the output holds the features from `count` events ago aligned with the target
value that is known at time `t`. The pairing is fully causal: it lags `X`,
never leads `y`, so nothing here peeks into the future.

<!-- HELP_END -->

## Causal framing

The target `y` must itself be causal, meaning its value at index `t` is fully
determined by data available at or before `t`. A trailing rolling sum or
exponential mean satisfies this; a centered window does not. `forecast_pairs`
shifts the features backward in time by `count` events (or by `duration`
index-units), then aligns the shifted features to the target clock. The result
is a training set where knowing `X` at time `t - count` lets a model predict
`y` at time `t`.

## Parameters

- `X`: feature array (1-D or 2-D) for `count=` mode, or a `(values, index)`
  tuple for `duration=` mode, whose values may also be 1-D or 2-D.
- `y`: target array, one column (`(T,)` or `(T, 1)`), for `count=` mode, or a
  `(values, index)` tuple for `duration=` mode. Must be causal.
- `count=` (int): shift `X` by this many events. Event-based; no index needed.
  Mutually exclusive with `duration=`.
- `duration=` (numeric): shift `X`'s index by this many index-units via
  `Delay`. Time-based; requires an explicit index on both `X` and `y`.
  Mutually exclusive with `count=`.
- `dropna=False`: if `True`, drop rows where `X_shifted` or `y` contains NaN.
  The first `count` rows of `X_shifted` are NaN (the warmup period where no
  lagged feature exists yet); `dropna=True` removes them automatically.

## Returns

A two-tuple `(X_shifted, y)`:

- `X_shifted`: the features delayed by the forecast horizon (`X[t-count]` at row `t`).
  It has the rank of `X`: a 2-D `X` with `K` columns gives a `(rows, K)` array.
- `y`: the target, passed through unchanged apart from any rows removed by `dropna`.

The rows stay in time order. To map a row back to its time, keep your own index
alongside `X` and `y`: a pandas index, or the index you already pass in `duration=`
mode.

## One-of constraint

Exactly one of `count=` or `duration=` must be supplied. Passing both or
neither raises `ValueError`. `count` is from 1 to 2**31 - 1 and `duration` is
at least 0 (a negative duration would pair a target with later features); a
value out of range raises `ValueError`. Both are numbers: a string raises
`TypeError`, and a float is truncated, as `Delay` truncates its duration.
`dropna` is a bool. `X` is `(T,)` or `(T, K)`, and `y` is one column; another
shape raises `ValueError`.

## Regimes

`count=` mode runs identically in all three regimes and produces byte-identical
output across them. The function dispatches on the input type:

- **Eager** (numpy arrays): call directly; returns `(X_shifted, y)` as numpy arrays.
- **Graph** (Node inputs): pass `Input(...)` nodes; returns a Node. Wrap it in a
  `Pipeline` to run. The output node produces a 2-column array with columns
  `[X_shifted, y]`; index 0 is `X_shifted`, index 1 is `y`. With a `K`-column
  `X`, the row is `[X_shifted (K columns), y]`.
- **Lazy** (generators of `(value, index)` events): returns a lazy iterator of
  `((xs_val, y_val), index)` events.

`duration=` mode runs in all three regimes with byte-identical output across them.
The function dispatches on the input type:

- **Eager** `(duration=)`: pass `X` and `y` as `(values, index)` tuples. Returns
  `(X_shifted, y)` as numpy arrays.
- **Graph** `(duration=)`: pass `Input(...)` nodes; returns a Node. The output node
  produces a 2-column array with columns `[X_shifted, y]`. Wrap in a `Pipeline`.
- **Lazy** `(duration=)`: pass generators of `(value, index)` events; returns a lazy
  iterator of `((xs_val, y_val), index)` events.

Every regime, and a live session of a `Pipeline` that holds the node, runs one
graph defined in C++. Count mode lags the rows of `X`, of any width, by `count`
events, as `Lag(count)` lags one column.
Duration mode is `Resample(Delay(duration)(X), clock=y, agg='last', fill='carry')`:
each `y` is paired with the last `X` whose index plus `duration` is at or before
the index of `y`, column by column: a NaN in a column of that row gives the
column's last non-NaN value. Both modes then pair the shifted features with
`y` through `CombineLatest`.

## JavaScript

`forecastPairs(X, y, { count | duration, dropna })` builds the same graph and
returns `[X_shifted, y]`, with the same values as Python. In count mode `X` is a
1-D array, a `[T, K]` NdArray or T rows, and `y` one column of the same length.
In duration mode `X` and `y` are `{ values, index }` streams. Given `Input` nodes,
it returns a Node for a `Pipeline`.

## Examples

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import RollingSum
    from screamer.supervised import forecast_pairs

    rng = np.random.default_rng(42)
    X = rng.standard_normal(200)

    # Build a causal target: sum of the next h events, but expressed as
    # a trailing RollingSum so it is causal at each step.
    h = 5
    y = RollingSum(h)(X)

    X_shifted, y_train = forecast_pairs(X, y, count=h, dropna=True)

    print("training rows:", len(X_shifted))
    print("X_shifted[:3]:", X_shifted[:3].round(4))
    print("y_train[:3]  :", y_train[:3].round(4))
```
