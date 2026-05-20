---
name: TRIMA
title: Triangular MA (TRIMA)
implementation_family: rolling
topics:
- trend
- smoothing
tags:
- trima
- triangular
- moving-average
short: 'Triangular MA: SMA of an SMA. Heavier center-weighting than WMA.'
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
---

# `TRIMA`

## Description

`TRIMA` (Triangular Moving Average) is a double-smoothed simple mean: an SMA of an SMA. The effective per-sample weights form a symmetric triangle (rising then falling), giving more weight to the centre of the window than the ends.

$$
\text{TRIMA}(x, n)[t] = \text{SMA}(\text{SMA}(x, n_\text{inner}), n_\text{outer})[t]
$$

with TA-Lib's window split:

| Total window `n` | `n_inner` | `n_outer` |
|---|---|---|
| odd | `(n + 1) / 2` | `(n + 1) / 2` |
| even | `n/2 + 1` | `n/2` |

In both cases `n_inner + n_outer - 1 == n`, so the effective triangular weighting spans `n` samples.

## Parameters

- `window_size` (int, positive). Total triangle width.

*NaN handling*: NaN values should be preprocessed.

## Implementation Details

### Algorithm

Pure composition of two chained `detail::RollingMean` instances. Both run with `start_policy="expanding"` so that the inner doesn't emit `NaN` (which would poison the outer's running sum permanently). `TRIMA` itself enforces strict warmup by counting samples and emitting `NaN` until `n` samples have been processed.

### Complexity

* Time complexity: `O(1)` per step (two `RollingMean` updates).
* Space complexity: `O(window_size)`.

## Examples

### Usage example

```python
import numpy as np
from screamer import TRIMA, RollingMean

x = np.cumsum(np.random.randn(100))
n = 10

# Direct
ours = TRIMA(n)(x)

# Algorithmically equivalent composition (post-warmup, the test suite
# verifies bit-equality)
n_inner, n_outer = (n // 2 + 1, n // 2) if n % 2 == 0 else ((n + 1) // 2, (n + 1) // 2)
inner = RollingMean(n_inner, "expanding")(x)
outer = RollingMean(n_outer, "expanding")(inner)
np.testing.assert_allclose(ours[n - 1:], outer[n - 1:], atol=1e-12)
```

<!-- HELP_END -->

## Reference

Equivalent to TA-Lib's `TRIMA`. Validated in `tests/test_moving_averages.py` against the explicit two-`RollingMean` composition for several window sizes (both even and odd).
