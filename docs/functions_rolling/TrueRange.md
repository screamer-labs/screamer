---
name: TrueRange
title: True Range (Wilder)
implementation_family: rolling
topics:
- volatility
tags:
- wilder
- true-range
- tr
- talib
- hlc
short: Per-bar true range accounting for overnight gaps (Wilder, 1978).
inputs: 3
outputs: 1
parameters: []
---

# `TrueRange`

## Description

The per-bar true-range quantity that accounts for overnight gaps:

$$
\text{TR}[t] = \max\big(\ H - L,\ \ |H - C_{t-1}|,\ \ |L - C_{t-1}|\ \big)
$$

**3-input, 1-output** on `(high, low, close)`. The first sample returns `NaN` (no previous
close). Otherwise stateless. Bit-exact to `talib.TRANGE`.

<!-- HELP_END -->
