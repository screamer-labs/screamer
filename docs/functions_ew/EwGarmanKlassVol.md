---
name: EwGarmanKlassVol
title: EW Garman-Klass olatility
implementation_family: ew
topics:
- volatility
tags:
- garman-klass
- range-based
- ohlc
- vol
- ew
short: Vol form of the Garman-Klass range-based volatility estimator (OHLC).
inputs: 4
outputs: 1
parameters:
- name: com
  type: float
  default: null
  description: Center of mass.
- name: span
  type: float
  default: 20.0
  description: Span.
- name: halflife
  type: float
  default: null
  description: Halflife.
- name: alpha
  type: float
  default: null
  description: Smoothing parameter directly.
nan_policy: ignore
---

# `EwGarmanKlassVol`

## Description

The Garman-Klass (1980) range-based volatility estimator. Per-bar variance contribution:

$$
\sigma^2_\text{GK}[t] = \tfrac{1}{2}\big(\ln H/L\big)^2 - (2\ln 2 - 1)\big(\ln C/O\big)^2
$$

This expression is averaged with a exponentially-weighted mean to form the estimator. The `Vol`
variant returns `sqrt(Var)` (bit-exact via the same internal state).

**4-input, 1-output** on `(open, high, low, close)`. ~7.4x more statistically efficient
than close-to-close `RollingStd` *under the model's assumptions* (zero drift, no overnight
gaps).


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->
