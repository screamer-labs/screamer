---
name: RollingMaxDrawdown
title: Rolling maximum drawdown
implementation_family: fin
topics:
- risk
tags:
- drawdown
- max-drawdown
- rolling
short: Worst peak-to-trough drawdown inside a trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 252
  min: 2
  description: Trailing-window length (252 = one trading year, default).
nan_policy: ignore
---

# `RollingMaxDrawdown`

## Description

The worst peak-to-trough loss observed inside the last `window_size` bars. Different from
`MaxDrawdown` (which is the worst loss EVER since reset).

## Implementation

Maintains a circular buffer of the last `w` prices and, on each step, sweeps the buffer
tracking a within-window running peak and the worst drawdown from that peak. **O(window_size)
per step** -- there is no cheap amortised algorithm for the standard definition because the
in-window peak can sit anywhere in the window.

If you want the cheaper "current drawdown vs. rolling-window peak" approximation, compose it
directly:


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Implementation

```python
rolling_dd = price / RollingMax(window)(price) - 1
```

<!-- HELP_END -->
