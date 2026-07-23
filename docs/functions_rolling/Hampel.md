---
name: Hampel
title: Hampel filter
implementation_family: rolling
topics:
- filtering
- outliers
tags:
- outlier
- despike
- hampel
- robust
- rolling
short: Robust Hampel despiker, replace samples far from the window median (in MAD units).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Trailing-window length.
- name: n_sigma
  type: float
  default: 3.0
  min: 0
  description: Threshold in robust standard deviations (1.4826 * MAD).
- name: output
  type: str
  default: cleaned
  enum:
  - cleaned
  - flag
  - nan
  description: '"cleaned" replaces outliers with the median, "flag" emits 1.0 at outliers (else 0.0), "nan" replaces outliers with NaN.'
- name: start_policy
  type: str
  default: strict
  description: Warmup handling before the window is full (strict, expanding, or zero).
nan_policy: ignore
---

# `Hampel`

## Description

`Hampel` is the canonical robust despiker (the **Hampel filter** or Hampel
identifier), in its causal trailing-window form. Over the trailing window it computes
the median `m` and the median absolute deviation `MAD`, and flags a sample as an
outlier when

$$ \bigl| x_t - m_t \bigr| > \texttt{n\_sigma} \times 1.4826 \times \text{MAD}_t $$

(the factor `1.4826` makes the MAD a Gaussian-consistent standard-deviation estimate).
Because it uses the median and MAD rather than the mean and standard deviation, a few
spikes cannot drag the centre or inflate the scale. A flagged sample is replaced by
the window median, and the replacement, not the raw outlier, is fed back into the
window so a burst of spikes cannot pollute later scale estimates.

It is strictly causal (trailing window only). For strongly non-stationary signals see [`ImpulseClip`](ImpulseClip.md),
which detects on the trend-free first difference.

*Parameters*:
- **`window_size`**: *(int)* Trailing-window length. Must be positive.
- **`n_sigma`**: *(float)* Detection threshold in robust standard deviations. Larger
  values flag fewer samples. Typical value `3.0`.
- **`output`**: *(str)* What to return:
  - `"cleaned"` (default): the cleaned signal, outliers replaced by the median.
  - `"flag"`: an outlier flag, `1.0` where a sample is flagged, else `0.0`.
  - `"nan"`: the input with flagged samples replaced by `NaN`.
- **`start_policy`**: Warmup handling before `window_size` samples are available
  (`"strict"`, `"expanding"`, or `"zero"`).

A degenerate note: when the window is exactly constant the MAD is zero, so no sample is
flagged (there is no scale to measure against).

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip
that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite
samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```python
import numpy as np
from screamer import Hampel

rng = np.random.default_rng(0)
x = np.sin(np.linspace(0, 8, 300)) + 0.05 * rng.standard_normal(300)
x[150] += 6.0                       # a spike

cleaned = Hampel(window_size=21, n_sigma=3.0)(x)                      # spike removed
flags = Hampel(window_size=21, n_sigma=3.0, output="flag")(x)         # 1.0 at the spike
```

<!-- HELP_END -->

## Implementation Details

O(W) per step: each step copies the trailing window, computes the median and the
median absolute deviation with `std::nth_element`, and applies the threshold. Detected
outliers are written back into the window as the median so they do not bias later
medians. Strictly causal.
