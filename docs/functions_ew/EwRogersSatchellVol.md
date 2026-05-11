---
name: EwRogersSatchellVol
title: EW Rogers-Satchell olatility
implementation_family: ew
topics:
- volatility
tags:
- rogers-satchell
- range-based
- ohlc
- drift-robust
- vol
- ew
short: Vol form of the Rogers-Satchell drift-robust range-based estimator.
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
---

# `EwRogersSatchellVol`

## Description

The Rogers-Satchell (1991) range-based volatility estimator. Per-bar variance contribution:

$$
\sigma^2_\text{RS}[t] = \ln\tfrac{H}{C}\ \ln\tfrac{H}{O} + \ln\tfrac{L}{C}\ \ln\tfrac{L}{O}
$$

This expression is averaged with a exponentially-weighted mean to form the estimator. The `Vol`
variant returns `sqrt(Var)` (bit-exact via the same internal state).

**4-input, 1-output** on `(open, high, low, close)`. Slightly less efficient (~6x vs
close-to-close) than Garman-Klass but **drift-robust** — works correctly when the underlying
drift is non-zero, a much more realistic assumption for trending markets.

<!-- HELP_END -->
