---
name: segment_shuffle
title: segment_shuffle
kind: function
short: Copies of a series with its runs of equal value in random order, for randomization
  tests.
tags:
- shuffle
- runs
- randomization test
- monte carlo
see_also:
- position_test
- circular_shift
- permute_returns
topics:
- significance
---

# `segment_shuffle`

Returns `n` copies of a series as a `(T, n)` array. The series is split into
runs of equal value, and each copy writes the runs back in a random order in
which no two neighbouring runs have the same value. For a target-position
series, each trade and each flat stretch is one run, so every copy holds exactly
the same trades, with the same sides, sizes, and holding periods, at random
times, and changes position as often as the original.

Two neighbouring runs of the same value would merge into one longer trade, and
a copy with fewer trades pays less cost, which would bias a test with costs.
When the series alternates between two values with nothing in between, such as
a long/short signal with no flat stretches, the runs can only alternate, so the
copies keep the alternation and move the run lengths.

The leading NaN warmup stays in place. A NaN inside the series is a run of its
own, like any other value.

The function is batch-only and runs in C++. Each column has its own generator,
seeded from `seed` and the column number, so the same seed gives the same
columns on every platform. [`position_test`](position_test.md) uses it with
`method="segments"`.

<!-- HELP_END -->

## Signature

`segment_shuffle(x, n, seed=0)`

## Examples

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import segment_shuffle

    target = np.array([np.nan, 1, 1, 1, 0, 0, -1, -1, 0, 1])
    print(segment_shuffle(target, 3, seed=0))
```
