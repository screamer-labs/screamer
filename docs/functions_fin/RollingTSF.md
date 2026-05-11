---
name: RollingTSF
title: Rolling Time-Series Forecast (TSF)
implementation_family: fin
topics:
- regression
tags:
- tsf
- forecast
- regression
- talib
short: Linear regression of y on time, projected one step ahead. TA-Lib's TSF.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
---

# `RollingTSF`

## Description

TA-Lib's *Time-Series Forecast*: fits `y = slope · t + intercept` on the trailing window
`{(0, y_{t-w+1}), …, (w-1, y_t)}` and returns the line evaluated at the *next* bar (local
`t = w`):

$$
\text{TSF}[t] = \text{intercept} + \text{slope} \cdot w
$$

1→1. Bit-exact match to `talib.TSF(real, timeperiod)`. Composes two `detail::RollingSum`
buffers + precomputed time-axis constants. O(1) per step.

<!-- HELP_END -->
