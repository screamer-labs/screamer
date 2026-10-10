---
name: permute_returns
title: permute_returns
kind: function
short: Copies of a price series with its returns in random order.
topics:
- backtesting
---

# `permute_returns`

Returns `n` randomized copies of a price series as a `(T, n)` array. The steps
of the series (log returns with `log=True`, otherwise differences) are shuffled
and the path is rebuilt from the first valid value. Every copy keeps the first
and last values and the distribution of returns, but not their order. The NaN
warmup prefix stays in place.

Use `log=True` (the default) for prices, which must then be positive, and
`log=False` for series that can cross zero, such as a spread. The function is
batch-only and runs in C++; the same `seed` gives the same copies on every
platform. [`strategy_test`](strategy_test.md) uses it with `method="returns"`.

<!-- HELP_END -->

## Signature

`permute_returns(x, n, log=True, seed=0)`

## Example

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import permute_returns

    price = np.array([100.0, 101.0, 100.5, 102.0, 103.0, 102.5])
    print(np.round(permute_returns(price, 3, seed=0), 2))
```
