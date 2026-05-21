---
name: ButterHighpass
title: Butterworth high-pass filter
implementation_family: signal
topics:
- signal-processing
tags:
- butterworth
- iir
- highpass
short: General-order IIR Butterworth high-pass filter (rejects low frequencies).
inputs: 1
outputs: 1
parameters:
- name: order
  type: int
  default: 2
  min: 1
  description: Filter order.
- name: cutoff_freq
  type: float
  default: 0.1
  min: 0.0
  max: 0.5
  description: Normalised cutoff.
nan_policy: ignore
---

# `ButterHighpass`

## Description

Digital high-pass Butterworth IIR filter, designed via the standard analog-prototype + `lp2hp` + bilinear pipeline.

$$
\text{ButterHighpass}(N, W_c)
$$

- `order` (int, $\ge 1$): filter order. Higher order = steeper roll-off, larger group delay.
- `cutoff_freq` (float in $(0, 1)$): normalised cutoff. `1` is the Nyquist frequency (matches `scipy.signal.butter`'s `Wn` convention).


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->

## Implementation Details

Bit-exact match to `scipy.signal.butter(order, cutoff, btype='highpass')` + `scipy.signal.lfilter`. Verified to ~1e-12 in `tests/test_signal.py`.

```python
from screamer import ButterHighpass
hp = ButterHighpass(order=4, cutoff_freq=0.1)
out = hp(noisy_signal)
```

Uses the same `IIRFilter` engine as the existing `Butter` low-pass class -- O(order) per step.
