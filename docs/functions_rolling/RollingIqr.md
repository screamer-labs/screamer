---
name: RollingIqr
title: Rolling interquartile range
implementation_family: rolling
topics:
- statistics
tags:
- iqr
- quartile
- rolling
short: Q3 minus Q1 over the trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
---

# `RollingIqr`

## Description

`RollingIqr` computes the rolling **inter-quartile range**:

$$
\text{IQR}[t] = Q_{0.75}[t] - Q_{0.25}[t]
$$

A robust spread measure: discards the top and bottom 25% of the window, so it is unaffected by single-point outliers. Useful as the denominator of a robust z-score, or for outlier-resistant volatility heuristics.

*Parameters*: `window_size` (int, positive).

*NaN handling*: NaN values should be preprocessed.

## Examples

### Usage example

```python
import numpy as np
import pandas as pd
from screamer import RollingIqr

rng = np.random.default_rng(0)
x = rng.standard_normal(500)
iqr = RollingIqr(30)(x)

# Validate against pandas (post-warmup, to ~1e-12)
ref = (
    pd.Series(x).rolling(30).quantile(0.75)
    - pd.Series(x).rolling(30).quantile(0.25)
).to_numpy()
np.testing.assert_allclose(iqr[29:], ref[29:], atol=1e-12)
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

A single `OrderStatisticTree` (the same primitive `RollingQuantile` uses) is queried twice per step -- once at the `0.25 * (n - 1)` position and once at `0.75 * (n - 1)` -- with linear interpolation between adjacent order statistics, identical to `RollingQuantile`'s formula.

### Why not just two `RollingQuantile` instances?

Composing as `RollingQuantile(w, 0.75)(x) - RollingQuantile(w, 0.25)(x)` would work but use two independent OSTs. The dedicated implementation has:

| | Two `RollingQuantile` | `RollingIqr` |
|---|---|---|
| Memory | 2 trees (≈ 2W nodes) | 1 tree (≈ W nodes) |
| Inserts/erases per step | 2 + 2 | 1 + 1 |
| Asymptotic complexity | `O(log W)` | `O(log W)` |

Same asymptotic complexity, **half the memory and half the work** per step. Validated in tests against the composition reference (post-warmup) to floating-point precision.

### Complexity

* Time complexity: `O(log W)` per step.
* Space complexity: `O(window_size)`.

## Reference

Equivalent to `pandas.Series.rolling(w).quantile(0.75) - pandas.Series.rolling(w).quantile(0.25)`.
