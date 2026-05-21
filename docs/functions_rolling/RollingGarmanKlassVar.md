---
name: RollingGarmanKlassVar
title: Rolling Garman-Klass varariance
implementation_family: rolling
topics:
- volatility
tags:
- garman-klass
- range-based
- ohlc
- var
- rolling
short: Var form of the Garman-Klass range-based volatility estimator (OHLC).
inputs: 4
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Smoothing window.
nan_policy: ignore
---

# `RollingGarmanKlassVar`

## Description

The Garman-Klass (1980) range-based volatility estimator. Per-bar variance contribution:

$$
\sigma^2_\text{GK}[t] = \tfrac{1}{2}\big(\ln H/L\big)^2 - (2\ln 2 - 1)\big(\ln C/O\big)^2
$$

This expression is averaged with a rolling mean over `window_size` bars to form the estimator. The `Vol`
variant returns `sqrt(Var)` (bit-exact via the same internal state).

**4-input, 1-output** on `(open, high, low, close)`. ~7.4x more statistically efficient
than close-to-close `RollingStd` *under the model's assumptions* (zero drift, no overnight
gaps).


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->
