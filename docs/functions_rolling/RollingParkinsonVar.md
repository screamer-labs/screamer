---
name: RollingParkinsonVar
title: Rolling Parkinson varariance
implementation_family: rolling
topics:
- volatility
tags:
- parkinson
- range-based
- hl
- var
- rolling
short: Var form of the Parkinson range-based volatility estimator (uses high & low).
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Smoothing window for the per-bar estimator.
nan_policy: ignore
---

# `RollingParkinsonVar`

## Description

The Parkinson (1980) range-based volatility estimator. Per-bar variance contribution:

$$
\sigma^2_\text{Parkinson}[t] = \frac{1}{4 \ln 2}\ \big(\ln H/L\big)^2
$$

This expression is then averaged with a rolling mean over `window_size` bars to form the estimator. The
`Vol` variant returns `sqrt(Var)`; the two are bit-exact via the same internal state.

**2-input, 1-output** on `(high, low)`. ~5x more statistically efficient than
close-to-close `RollingStd` *under the model's assumptions* (zero drift, no overnight gaps).


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->
