---
name: RollingArgmax
title: Rolling argmax
implementation_family: rolling
topics:
- statistics
tags:
- argmax
- rolling
short: Window-offset of the trailing-window maximum (TA-Lib MAXINDEX).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
---

# `RollingArgmax`

## Description

`RollingArgmax` returns the *position* (within the current window) of the rolling maximum value, rather than the maximum itself. Convention: **0 = oldest sample in the window**, **window_size−1 = newest**. Matches `numpy.argmax` applied to the trailing window slice and `pandas.Series.rolling(w).apply(np.argmax)`.

*Parameters*: `window_size` (int, positive).

*NaN handling*: NaN values should be preprocessed.

<!-- HELP_END -->

## Usage Example

```python
import numpy as np
from screamer import RollingArgmax, RollingMax

x = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5], dtype=float)
RollingArgmax(5)(x)        # window offsets, 0 = oldest in window
RollingMax(5)(x)           # corresponding maxima
```

## Implementation Details

### Algorithm

Same monotonic-deque primitive (`detail::MaxDeque`) used by `RollingMax`, `RollingMinMax`, and `RollingRange`. Each deque entry stores `(value, absolute_sample_index)`; the front entry is always the current rolling maximum, and we expose its window offset.

### Complexity

* Time complexity: `O(1)` amortised per step.
* Space complexity: `O(window_size)`.

## Reference

Equivalent to `pandas.Series.rolling(w).apply(np.argmax, raw=True)` for samples after warmup. Also equivalent to TA-Lib's `MAXINDEX`.
