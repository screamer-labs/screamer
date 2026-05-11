---
name: ATR
title: Average True Range (ATR)
implementation_family: rolling
topics:
- volatility
tags:
- wilder
- atr
- true-range
- talib
- hlc
short: Wilder-smoothed average of TrueRange.
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Wilder smoothing period (Wilder's original choice is 14).
---

# `ATR`

## Description

Wilder-smoothed rolling average of `TrueRange`:

$$
\begin{aligned}
\text{ATR}[w] &= \frac{1}{w} \sum_{i=1}^{w} \text{TR}[i] \quad\text{(SMA seed)} \\
\text{ATR}[t] &= \frac{(w - 1) \cdot \text{ATR}[t - 1] + \text{TR}[t]}{w} \quad\text{for } t > w
\end{aligned}
$$

**3-input, 1-output** on `(high, low, close)`. First valid output at sample index
`window_size`. Matches `talib.ATR` bit-exactly post-warmup.

<!-- HELP_END -->
