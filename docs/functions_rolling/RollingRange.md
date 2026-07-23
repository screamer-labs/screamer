---
name: RollingRange
title: Rolling range
implementation_family: rolling
topics:
- statistics
- volatility
tags:
- range
- min
- max
- rolling
short: Trailing-window max minus min.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
nan_policy: ignore
---

# `RollingRange`

## Description

`RollingRange` returns the rolling range, i.e. `max − min` over the trailing window. Common in volatility heuristics, breakout detection, and as the bandwidth of channel-style indicators (Donchian, etc.).

*Parameters*: `window_size` (int, positive).

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```python
import numpy as np
from screamer import RollingRange, RollingMinMax

x = np.cumsum(np.random.normal(size=200))

# Direct
range_x = RollingRange(20)(x)

# Algorithmically equivalent composition
mm = RollingMinMax(20)(x)
np.testing.assert_array_equal(range_x, mm[:, 1] - mm[:, 0])
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

`RollingRange` holds a `detail::MinDeque` and a `detail::MaxDeque` -- the same primitive used by `RollingMin`, `RollingMax`, and `RollingMinMax`. On each step both deques are updated and the difference between their fronts is returned.

This is *deliberately the same algorithm* as `RollingMinMax` followed by a subtract; the only thing the dedicated class saves is the tuple-allocation of the 1->2 dispatcher per step, which is a constant-factor win at best. Memory and compute order are identical to `RollingMinMax`.

### Complexity

* Time complexity: `O(1)` amortised per step.
* Space complexity: `O(window_size)` (two deques, each bounded by the window).

## Reference

`RollingMax(w)(x) - RollingMin(w)(x)` and `pandas.Series.rolling(w).max() - pandas.Series.rolling(w).min()` are both algorithmically equivalent (post-warmup, to floating-point precision).
