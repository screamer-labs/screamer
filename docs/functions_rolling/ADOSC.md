---
name: ADOSC
title: Chaikin A/D Oscillator (ADOSC)
implementation_family: rolling
topics:
- volume
- oscillator
tags:
- adosc
- chaikin
- oscillator
- talib
- ohlcv
short: Difference of fast and slow EMA of the Accumulation/Distribution line.
inputs: 4
outputs: 1
parameters:
- name: fast
  type: int
  default: 3
  min: 2
  description: Fast EMA period.
- name: slow
  type: int
  default: 10
  min: 2
  description: Slow EMA period.
---

# `ADOSC`

## Description

Chaikin A/D Oscillator: difference of two EMAs of the `AD` line.

$$
\text{ADOSC}[t] = \text{EMA}(\text{AD},\ \text{fast})[t] - \text{EMA}(\text{AD},\ \text{slow})[t]
$$

**4-input, 1-output** on `(high, low, close, volume)`. Default `(fast=3, slow=10)` matches
TA-Lib.

The underlying EMA is `screamer.EwMean` (pandas `adjust=True`), so `ADOSC` inherits the
same documented divergence from TA-Lib's `ADOSC` as `DEMA`/`TEMA`/`MACD`/`TRIX`. The class
matches the explicit pandas-composition reference bit-exactly. See `docs/conventions.md`
for the divergence detail.

<!-- HELP_END -->
