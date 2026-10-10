---
name: permute_bars
title: permute_bars
kind: function
short: Bar-permuted copies of an OHLC series that keep its return distribution but no serial
  structure.
tags:
- permutation
- shuffle
- monte carlo
- synthetic data
- ohlc
see_also:
- permute_returns
- strategy_test
- circular_shift
- segment_shuffle
references:
- Masters, T. (2020). *Permutation and Randomization Tests for Trading System Development*.
  Independently published.
topics:
- significance
---

# `permute_bars`

Returns `n` randomized copies of an OHLC series as four `(T, n)` arrays: open,
high, low, and close. `mode` sets what is shuffled:

- `"log"` (default): each bar after the first splits into a gap (its open
  relative to the previous close) and an intrabar shape (its high, low, and close
  relative to its open), as log ratios. The gaps and the shapes are shuffled with
  two independent permutations, and the path is rebuilt from the first bar
  (Masters 2020). Prices must be positive.
- `"diff"`: the same with differences instead of log ratios, for prices that can
  be zero or negative (spread bars, rates).
- `"level"`: whole bars are shuffled, each keeping its own open, high, low, and
  close. Use it for series that are stationary in level.

In `"log"` and `"diff"` mode every copy keeps the first bar, the last close, the
distribution of gaps, the distribution of intrabar shapes, and valid bars (the
low is at most the open and the close, the high at least both). Any serial
structure, such as momentum, mean reversion, or volatility clustering, is gone.
That makes the copies a null market for [`strategy_test`](strategy_test.md).

With `volume`, a fifth array holds the volume of the bar each row's shape came
from. `volume_mode="keep"` (default) moves it unchanged. `"notional"` rescales it
so the traded amount, volume times the typical price (high + low + close) / 3,
stays that of its source bar: where a copy trades at twice the price, the volume
halves. In `"level"` mode the bars move whole, so both give the same volume.

The function is batch-only and runs in C++; the same `seed` gives the same
copies on every platform.

<!-- HELP_END -->

## Signature

`permute_bars(open, high, low, close, n, mode="log", seed=0, volume=None,
volume_mode="keep")`

## Examples

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import permute_bars

    close = np.array([100.0, 101.0, 100.5, 102.0, 103.0, 102.5])
    open_ = np.r_[100.0, close[:-1]]
    high, low = np.maximum(open_, close) + 0.5, np.minimum(open_, close) - 0.5
    o, h, l, c = permute_bars(open_, high, low, close, 3, seed=0)
    print(np.round(c, 2))
```
