---
name: HullMA
title: Hull MA
implementation_family: rolling
topics:
- trend
- smoothing
tags:
- hull
- hullma
- moving-average
short: 'Hull''s responsive MA: WMA(2*WMA(n/2) - WMA(n), sqrt(n)).'
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

# `HullMA`

## Description

`HullMA` (Alan Hull, 2005) is a low-lag responsive moving average defined entirely in terms of `WMA`:

$$
\text{HullMA}(x, n)[t] = \text{WMA}\Big( 2 \cdot \text{WMA}(x, n/2) - \text{WMA}(x, n),\ \sqrt{n} \Big)[t]
$$

with integer floor on the inner window arguments: `n_half = n // 2` and `n_sqrt = floor(sqrt(n))`.

The construction subtracts a slow `WMA` from twice a fast `WMA` (anticipating the trend, similar to `DEMA`'s linear extrapolation), then smooths the result with a much shorter `WMA`. The output tracks the price closely with markedly less lag than a plain SMA / EMA / WMA of comparable window.

## Parameters

- `window_size` (int, **at least 4**). The construction degenerates below `n=4` because `n_half` must be `>= 2` and `floor(sqrt(n))` must be `>= 2`.

*NaN handling*: NaN values should be preprocessed.

## Implementation Details

### Algorithm

Pure composition of three `WMA` instances:

1. `wma_half_` -- `WMA(n // 2, "expanding")` on the input `x`.
2. `wma_full_` -- `WMA(n, "expanding")` on the input `x`.
3. `wma_outer_` -- `WMA(floor(sqrt(n)), "expanding")` on `2*wma_half - wma_full`.

The inner WMAs run with `start_policy="expanding"` so they never emit `NaN` (which would poison the outer's state). `HullMA` enforces strict warmup itself by counting samples and emitting `NaN` until `n + floor(sqrt(n)) - 1` samples have been processed.

### Complexity

* Time complexity: `O(1)` per step (three `WMA` updates).
* Space complexity: `O(window_size)` (dominated by the longest internal `WMA`).


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```python
import numpy as np
from screamer import HullMA, WMA

x = np.cumsum(np.random.randn(200))
n = 16

# Direct
ours = HullMA(n)(x)

# Algorithmically equivalent composition (post-warmup; the test suite
# verifies bit-equality)
n_half = n // 2
n_sqrt = int(np.sqrt(n))
w_half = WMA(n_half, "expanding")(x)
w_full = WMA(n,      "expanding")(x)
w_outer = WMA(n_sqrt, "expanding")(2*w_half - w_full)
warmup = n + n_sqrt - 1
np.testing.assert_allclose(ours[warmup - 1:], w_outer[warmup - 1:], atol=1e-12)
```

<!-- HELP_END -->

## Reference

Standard Hull MA definition. Validated in `tests/test_moving_averages.py` against the explicit three-`WMA` composition for several window sizes.
