---
name: ImpulseClip
title: Impulse clip
implementation_family: rolling
topics:
- outliers
tags:
- outlier
- despike
- impulse
- glitch
- robust
- rolling
short: Causal impulse remover, detects spikes on the trend-free first difference.
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
  default: 4.0
  min: 0
  description: Threshold in robust standard deviations (1.4826 * MAD of the differences).
- name: output
  type: int|null
  default: null
  enum:
  - 0
  - 1
  - 2
  - null
  description: 0 = cleaned signal, 1 = outlier flag, 2 = outliers as NaN. None = cleaned.
- name: start_policy
  type: str
  default: strict
  description: Warmup handling before the window is full (strict, expanding, or zero).
nan_policy: ignore
---

# `ImpulseClip`

## Description

`ImpulseClip` removes impulse spikes (glitches) from a signal that may be strongly
non-stationary. A spike is a large, isolated jump: hard to separate from the signal's
own swing in the raw level, but obvious in the sample-to-sample change. So detection
runs on the first difference, whose scale is trend-free:

$$ d_t = x_t - x_{t-1}, \qquad
   \text{outlier if } |d_t| > \texttt{n\_sigma} \times 1.4826 \times \text{MAD}(d)_t, $$

where `MAD(d)` is the rolling median absolute deviation of the differences. A flagged
sample is replaced by the trailing median of the values.

It is strictly causal with zero latency, so batch and streaming results are identical.
Two consequences of detecting without looking ahead: an impulse is a `+/-` doublet in
the difference (a jump onto the spike and a jump back), so both the spike and its
return sample are flagged, replacing two consecutive samples (the second nudged to the
median); and a genuine level shift keeps its body, but its onset sample is clipped.
Preserving these would require a one-sample lookahead, which is deliberately not done.
For stationary data or multi-sample outliers, prefer [`Hampel`](Hampel.md).

*Parameters*:
- **`window_size`**: *(int)* Trailing-window length. Must be positive.
- **`n_sigma`**: *(float)* Detection threshold in robust standard deviations of the
  differences. Larger values flag fewer samples. Typical value `4.0`.
- **`output`**: *(optional, int)* What to return:
  - `0` (or `None`): the cleaned signal, outliers replaced by the median.
  - `1`: an outlier flag, `1.0` where a sample is flagged, else `0.0`.
  - `2`: the input with flagged samples replaced by `NaN`.
- **`start_policy`**: Warmup handling before `window_size` samples are available
  (`"strict"`, `"expanding"`, or `"zero"`).

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
from screamer import ImpulseClip

rng = np.random.default_rng(7)
t = np.linspace(0, 6 * np.pi, 400)
x = np.sin(t) + 0.3 * rng.standard_normal(t.size)   # oscillating + noise
x[[80, 210, 330]] += 4.0                            # spikes

cleaned = ImpulseClip(window_size=31, n_sigma=4.0)(x)   # spikes removed
```

<!-- HELP_END -->

## Implementation Details

O(W) per step. Two trailing windows are maintained: one of the values (for the
replacement median) and one of the first differences (for the robust scale). Each step
takes the median of the differences' absolute deviations with `std::nth_element` to set
the threshold, tests the current jump, and on a flag replaces the sample with the value
median (written back into the window to keep it clean). Strictly causal, zero latency.
