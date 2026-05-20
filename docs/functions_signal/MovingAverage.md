---
name: MovingAverage
title: FIR moving average (arbitrary taps)
implementation_family: signal
topics:
- signal-processing
- smoothing
tags:
- fir
- convolution
- moving-average
short: Finite-impulse-response filter with user-supplied taps.
inputs: 1
outputs: 1
parameters:
- name: taps
  type: float[]
  default:
  - 0.25
  - 0.5
  - 0.25
  description: FIR coefficients. Default is a 3-tap triangular kernel.
---

# `MovingAverage`

## Description

Finite-impulse-response (FIR) filter with arbitrary user-supplied tap coefficients:

$$
y[t] = \sum_{k=0}^{L-1} \text{taps}[k] \cdot x[t - k]
$$

`taps[0]` is the coefficient on the *current* sample; `taps[L-1]` is on the oldest. Pre-compute the coefficient vector with `numpy` (`np.hamming(n)`, `np.bartlett(n)`, `np.blackman(n)`, `np.kaiser(n, beta)`) or `scipy.signal.firwin` and pass it in. The user is responsible for any normalisation (e.g. dividing by `taps.sum()` for a unity-gain low-pass smoother).

*Parameters*:

- `taps` (sequence of float): the FIR coefficients.

*Warmup*: NaN for the first `len(taps) - 1` samples.

## Implementation Details

Circular buffer of the last `L = len(taps)` samples plus an in-order convolution sweep. **O(L) per step**.

## Examples

### Usage example

```python
import numpy as np
from screamer import MovingAverage

# Hamming-windowed low-pass smoother (unity gain).
taps = np.hamming(11)
taps /= taps.sum()
out = MovingAverage(list(taps))(signal)

# Uniform taps -> simple rolling mean (equivalent to RollingMean).
out = MovingAverage([1/7] * 7)(signal)
```

<!-- HELP_END -->

## Reference

Matches `numpy.convolve(x, taps, mode='full')[:len(x)]` post-warmup to floating-point precision. With uniform `1/n` taps it is bit-exact to `RollingMean(n)`.
