---
name: NATR
title: Normalised ATR (NATR)
implementation_family: rolling
topics:
- volatility
tags:
- wilder
- natr
- atr-normalized
- talib
- hlc
short: ATR scaled to a percentage of the current close.
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Wilder smoothing period.
---

# `NATR`

## Description

ATR scaled to a percentage of the current close:

$$
\text{NATR}[t] = 100\ \cdot\ \frac{\text{ATR}[t]}{C[t]}
$$

Useful for cross-instrument comparison (an ATR of \$1 on a \$10 stock is much larger than
the same ATR on a \$1000 stock).

**3-input, 1-output** on `(high, low, close)`. Bit-exact to `talib.NATR`.

<!-- HELP_END -->
