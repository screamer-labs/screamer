---
name: EwParkinsonVar
title: Exponentially-weighted Parkinson varariance
implementation_family: ew
topics:
- volatility
tags:
- parkinson
- range-based
- hl
- var
- ew
short: Var form of the Parkinson range-based volatility estimator (uses high & low).
inputs: 2
outputs: 1
parameters:
- name: com
  type: float
  default: null
  description: Center of mass. Exclusive with span/halflife/alpha.
- name: span
  type: float
  default: 20.0
  description: Span (default smoothing parameter).
- name: halflife
  type: float
  default: null
  description: Halflife. Exclusive with com/span/alpha.
- name: alpha
  type: float
  default: null
  description: Smoothing parameter directly.
---

# `EwParkinsonVar`

## Description

The Parkinson (1980) range-based volatility estimator. Per-bar variance contribution:

$$
\sigma^2_\text{Parkinson}[t] = \frac{1}{4 \ln 2}\ \big(\ln H/L\big)^2
$$

This expression is then averaged with a exponentially-weighted mean to form the estimator. The
`Vol` variant returns `sqrt(Var)`; the two are bit-exact via the same internal state.

**2-input, 1-output** on `(high, low)`. ~5x more statistically efficient than
close-to-close `RollingStd` *under the model's assumptions* (zero drift, no overnight gaps).

<!-- HELP_END -->
