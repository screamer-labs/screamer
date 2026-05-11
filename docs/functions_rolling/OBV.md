---
name: OBV
title: On-Balance Volume (OBV)
implementation_family: rolling
topics:
- volume
tags:
- obv
- granville
- talib
- pair
short: 'On-Balance Volume: signed cumulative volume by close-direction (Granville,
  1963).'
inputs: 2
outputs: 1
parameters: []
---

# `OBV`

## Description

On-Balance Volume (Granville, 1963):

$$
\text{OBV}[t] = \text{OBV}[t-1] + \text{sign}(C - C_{t-1})\ \cdot\ V[t]
$$

with seed `OBV[0] = volume[0]` (TA-Lib's convention). **2-input, 1-output** on
`(close, volume)`. Cumulative; no window.

Bit-exact to `talib.OBV` (0.0 difference).

<!-- HELP_END -->
