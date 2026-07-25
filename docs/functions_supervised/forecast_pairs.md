---
name: forecast_pairs
title: forecast_pairs
kind: function
short: Build a forecasting training set by aligning lagged features with a future causal target.
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
  tuple for `duration=` mode.
- `y`: target array (1-D) for `count=` mode, or a `(values, index)` tuple for
  `duration=` mode. Must be causal.
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
- `y`: the target, passed through unchanged apart from any rows removed by `dropna`.

The rows stay in time order. To map a row back to its time, keep your own index
alongside `X` and `y`: a pandas index, or the index you already pass in `duration=`
mode.

## One-of constraint

Exactly one of `count=` or `duration=` must be supplied. Passing both or
neither raises `ValueError`.

## Example

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
