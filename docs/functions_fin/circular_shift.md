---
name: circular_shift
title: circular_shift
kind: function
short: Randomly rotated copies of a series, for randomization tests.
tags:
- circular shift
- rotation
- randomization test
- monte carlo
see_also:
- position_test
- segment_shuffle
- permute_returns
topics:
- significance
---

# `circular_shift`

Returns `n` copies of a series as a `(T, n)` array. Each copy is the series
rotated by a random offset, so values that fall off the end wrap around to the
start. A rotation keeps every property of the series (its values, run lengths,
and autocorrelation) and changes only where it sits in time. This makes it the
standard way to build a null distribution for a trading signal: it destroys the
alignment between the signal and the market and nothing else.

The leading NaN warmup stays in place, and only the part after it is rotated.
The offset is uniform in `[min_shift, L - min_shift]`, where `L` is the length
after the warmup, so no copy is nearly aligned with the original. The default
`min_shift` is twice the mean run length, capped at a quarter of `L`.

The function is batch-only and runs in C++. Each column has its own generator,
seeded from `seed` and the column number, so the same seed gives the same
columns on every platform. [`position_test`](position_test.md) uses it with
`method="shift"`.

<!-- HELP_END -->

## Signature

`circular_shift(x, n, min_shift=None, seed=0)`

## Examples

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import circular_shift

    target = np.array([np.nan, 1, 1, 1, 0, 0, -1, -1, 0, 1])
    print(circular_shift(target, 3, min_shift=2, seed=0))
```
