---
name: RollingArgmin
title: Rolling argmin
implementation_family: rolling
topics:
- statistics
tags:
- argmin
- rolling
short: Window-offset of the trailing-window minimum (TA-Lib MININDEX).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
---

# `RollingArgmin`

## Description

`RollingArgmin` returns the *position* (within the current window) of the rolling minimum value, rather than the minimum itself. Convention: **0 = oldest sample in the window**, **window_size−1 = newest**. Matches `numpy.argmin` applied to the trailing window slice and `pandas.Series.rolling(w).apply(np.argmin)`.

*Parameters*: `window_size` (int, positive).

*NaN handling*: NaN values should be preprocessed (the deque comparison treats NaN as never beating an existing element).

<!-- HELP_END -->

## Usage Example

```python
import numpy as np
from screamer import RollingArgmin, RollingMin

x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], dtype=float)
RollingArgmin(5)(x)        # window offsets, 0 = oldest in window
RollingMin(5)(x)           # corresponding minima

# Example: how long ago was the running low? Convert offset to "steps ago":
steps_ago = (5 - 1) - RollingArgmin(5)(x).astype(int)
```

## Implementation Details

### Algorithm

`RollingArgmin` reuses the same monotonic-deque primitive used by `RollingMin`, `RollingMax`, `RollingMinMax`, and `RollingRange` (`detail::MinDeque`). Each deque entry stores `(value, absolute_sample_index)`; the front entry is always the current rolling minimum, and we expose its window offset.

### Complexity

* Time complexity: `O(1)` amortised per step.
* Space complexity: `O(window_size)`.

## Reference

Equivalent to `pandas.Series.rolling(w).apply(np.argmin, raw=True)` for samples after warmup. Pandas returns NaN during warmup; `RollingArgmin` returns the partial-window argmin from the very first sample.
