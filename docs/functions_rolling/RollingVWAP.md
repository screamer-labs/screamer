---
name: RollingVWAP
title: Rolling VWAP
implementation_family: rolling
topics:
- volume
tags:
- vwap
- volume-weighted
- ohlcv
- typical-price
short: Rolling volume-weighted average price (typical-price weighted).
inputs: 4
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
nan_policy: ignore
---

# `RollingVWAP`

## Description

Rolling volume-weighted average price using the typical price as the weighting basis:

$$
\text{TP}[t] = (\text{high} + \text{low} + \text{close}) / 3
\qquad
\text{VWAP}[t] = \dfrac{\sum_w \text{TP} \cdot \text{volume}}{\sum_w \text{volume}}
$$

**4-input, 1-output** on `(high, low, close, volume)`. Matches `pandas-ta-classic.vwap`.
For a *session-VWAP* (cumulative since some reset point), call `reset()` at the session
boundary.

First valid output at sample index `window_size - 1`. Composes two `detail::RollingSum`
instances; O(1) per step.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->
