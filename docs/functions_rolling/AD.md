---
name: AD
title: Accumulation/Distribution Line (Chaikin)
implementation_family: rolling
topics:
- cumulative
- volume
tags:
- ad
- accumulation-distribution
- chaikin
- talib
- ohlcv
short: Chaikin Accumulation/Distribution Line.
inputs: 4
outputs: 1
parameters: []
nan_policy: ignore
---

# `AD`

## Description

Chaikin Accumulation/Distribution Line:

$$
\text{CLV}[t] = \dfrac{(C - L) - (H - C)}{H - L}
\qquad
\text{AD}[t]  = \text{AD}[t-1] + \text{CLV}\ \cdot\ V[t]
$$

The "close location value" is in `[-1, +1]`: +1 means close at the high (full
accumulation), -1 at the low (full distribution). When `high == low` the CLV is undefined
and the AD line is unchanged (TA-Lib's convention).

**4-input, 1-output** on `(high, low, close, volume)`. Cumulative; no window. Bit-exact
to `talib.AD`.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->
