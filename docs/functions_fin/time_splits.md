---
name: time_splits
title: time_splits
kind: function
short: Train and test rows of time-ordered folds, leaving out the training rows whose labels reach into the test block.
tags:
- cross-validation
- time series split
- purging
- train test split
- out-of-sample
see_also:
- walk_forward
- forecast_pairs
- TripleBarrier
- TripleBarrierOHLC
references:
- 'Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, chapter 7. Wiley.'
topics:
- significance
---

# `time_splits`

Splits `n` time-ordered rows into folds for fitting and scoring a model. Each
fold has a block of test rows and trains only on rows before that block. The
result is a list of `(train, test)` pairs of row indices, which scikit-learn
accepts as `cv=`.

The folds are those of scikit-learn's `TimeSeriesSplit`. The `n_splits` test
blocks of `test_size` rows end at row `n`, so fold `f` tests on the rows
`[s, s + test_size)` with `s = n - (n_splits - f) * test_size`. Its training
rows are the rows before `s`, or the most recent `max_train_size` of them.

A label often covers several rows after its own row: the label of row `t`
from [`TripleBarrierOHLC`](TripleBarrierOHLC.md) uses the prices up to row
`t + holding[t]`. `label_length` gives that number of rows, one number for
every row or one value per row. A training row `t` is left out when
`t + label_length[t] >= s`, because its label uses prices from the test
block. A constant `label_length` equals the `gap` of `TimeSeriesSplit`.

A NaN or negative `label_length`, and a fold left with no training rows,
raise `ValueError`. A NaN usually marks a row whose label is not known yet;
drop those rows first, for example with `forecast_pairs(..., dropna=True)`.

<!-- HELP_END -->

## Signature

`time_splits(n, n_splits=5, *, test_size=None, label_length=0,
max_train_size=None)`

- `n`: the number of rows.
- `n_splits`: the number of folds, at least 2.
- `test_size`: the rows in each test block. The default is
  `n // (n_splits + 1)`.
- `label_length`: the rows after row `t` that its label covers, a number or an
  array of `n` values. The default 0 leaves no row out. A fractional value
  works: row `t` is left out when `t + label_length[t] >= s` holds for it.
- `max_train_size`: keep at most this many training rows per fold, the most
  recent ones that remain after leaving rows out. The default keeps all.

It returns a list of `n_splits` tuples `(train, test)` of int64 arrays.

## Examples

Twelve rows, three folds of three test rows, and labels that cover two rows
after their own row. Each fold leaves out its last two training rows.

```{eval-rst}
.. exec_code::

    from screamer import time_splits

    for train, test in time_splits(12, 3, label_length=2):
        print(train, test)
```

The `holding` output of a triple-barrier label as `label_length`, after
[`forecast_pairs`](../functions_supervised/forecast_pairs.md) moves each label
back to its entry row. A row just before a test block stays in when its
barrier is hit before the block starts, so the rows left out differ per fold.

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import TripleBarrier, forecast_pairs, time_splits

    rng = np.random.default_rng(0)
    price = 100 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01))
    tb = TripleBarrier(horizon=10, upper=0.02, lower=0.02)(price)
    _, holding = forecast_pairs(price, tb[:, 2], count=10, dropna=True)

    for train, test in time_splits(len(holding), 3, label_length=holding):
        print(f"train {train[0]}-{train[-1]} ({len(train)} rows), test {test[0]}-{test[-1]}")
```
