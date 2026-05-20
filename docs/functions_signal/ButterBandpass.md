---
name: ButterBandpass
title: Butterworth band-pass filter
implementation_family: signal
topics:
- signal-processing
tags:
- butterworth
- iir
- bandpass
short: General-order IIR Butterworth band-pass filter.
inputs: 1
outputs: 1
parameters:
- name: order
  type: int
  default: 2
  min: 1
  description: Filter order.
- name: low_cutoff
  type: float
  default: 0.05
  min: 0.0
  max: 0.5
  description: Lower cutoff frequency (normalised).
- name: high_cutoff
  type: float
  default: 0.2
  min: 0.0
  max: 0.5
  description: Upper cutoff frequency (normalised); must exceed low_cutoff.
---

# `ButterBandpass`

## Description

Digital band-pass Butterworth IIR filter. Produces a 2N-order filter from the order-N analog prototype via the `lp2bp` transformation.

$$
\text{ButterBandpass}(N, W_{\text{low}}, W_{\text{high}})
$$

- `order` (int, $\ge 1$): order of the analog prototype. Effective digital order is `2 * order`.
- `low_cutoff`, `high_cutoff` (floats in $(0, 1)$, with `low < high`): pass band as fractions of the Nyquist frequency.

Bit-exact match to `scipy.signal.butter(order, [low, high], btype='bandpass')` + `scipy.signal.lfilter`. Verified to ~1e-9 in `tests/test_signal.py`.

## Examples

### Description

```python
from screamer import ButterBandpass
bp = ButterBandpass(order=4, low_cutoff=0.1, high_cutoff=0.3)
out = bp(signal)
```

<!-- HELP_END -->
