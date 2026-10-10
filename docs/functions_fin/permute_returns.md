---
name: permute_returns
title: permute_returns
kind: function
short: Copies of a price series with its returns in random order.
tags:
- permutation
- shuffle
- monte carlo
- synthetic data
- returns
see_also:
- permute_bars
- strategy_test
- circular_shift
topics:
- significance
---

# `permute_returns`

Returns `n` randomized copies of a price series as a `(T, n)` array. `mode` sets
what is shuffled:

- `"log"` (default): the log returns. The path is rebuilt from the first valid
  value, so every copy keeps the first and last values and the distribution of
  returns, but not their order. Values must be positive.
- `"diff"`: the differences, rebuilt the same way. Use it for series that can be
  zero or negative, such as a spread.
- `"level"`: the values themselves. Use it for series that are stationary in
  level, such as a funding rate, where a rebuilt path would be meaningless.

The NaN warmup prefix stays in place.

With `volume`, the function returns `(prices, volume)`: the volume of the row
each step (or value) came from. `volume_mode="keep"` (default) moves it as it is;
`"notional"` rescales it so that volume times price stays that of its source
row: where a copy trades at twice the price, the volume halves. That keeps the
traded amount, which is what most volume-based signals respond to.

The function is batch-only and runs in C++; the same `seed` gives the same
copies on every platform. [`strategy_test`](strategy_test.md) uses it with
`method="returns"`.

<!-- HELP_END -->

## Signature

`permute_returns(x, n, mode="log", seed=0, volume=None, volume_mode="keep")`

## Examples

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import permute_returns

    price = np.array([100.0, 101.0, 100.5, 102.0, 103.0, 102.5])
    print(np.round(permute_returns(price, 3, seed=0), 2))
```
