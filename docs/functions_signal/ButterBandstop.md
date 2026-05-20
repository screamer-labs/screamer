---
name: ButterBandstop
title: Butterworth band-stop filter
implementation_family: signal
topics:
- signal-processing
tags:
- butterworth
- iir
- bandstop
- notch
short: General-order IIR Butterworth band-stop (notch) filter.
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
  description: Lower stop-band edge (normalised).
- name: high_cutoff
  type: float
  default: 0.2
  min: 0.0
  max: 0.5
  description: Upper stop-band edge (normalised); must exceed low_cutoff.
---

# `ButterBandstop`

## Description

Digital band-stop (notch) Butterworth IIR filter -- suppresses frequencies in `[low_cutoff, high_cutoff]`. Produces a 2N-order filter from the order-N analog prototype.

$$
\text{ButterBandstop}(N, W_{\text{low}}, W_{\text{high}})
$$

- `order` (int, $\ge 1$): order of the analog prototype.
- `low_cutoff`, `high_cutoff` (floats in $(0, 1)$, with `low < high`): stop band as fractions of the Nyquist frequency.

Bit-exact match to `scipy.signal.butter(order, [low, high], btype='bandstop')` + `scipy.signal.lfilter`. Verified to ~1e-9 in `tests/test_signal.py`.

## Examples

### Description

```python
from screamer import ButterBandstop
# 60 Hz notch on 1kHz-sampled data: Nyquist = 500 Hz, cutoffs ~58/500 .. 62/500
notch = ButterBandstop(order=2, low_cutoff=0.116, high_cutoff=0.124)
out = notch(signal)
```

<!-- HELP_END -->
