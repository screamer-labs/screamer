---
name: KeltnerChannels
title: Keltner Channels
implementation_family: rolling
topics:
- channels
tags:
- keltner
- channels
- envelope
- atr-based
short: 'Volatility-adapted envelope: EMA midline plus/minus a multiple of ATR.'
inputs: 3
outputs: 3
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Period for both the EMA midline and the ATR offset.
- name: num_atr
  type: float
  default: 2.0
  min: 0.0
  description: ATR multiplier for upper/lower offset.
nan_policy: ignore
---

# `KeltnerChannels`

## Description

Volatility-adapted envelope. The midline is an EMA of close; the upper/lower lines are
offset by a multiple of ATR:

$$
\begin{aligned}
\text{mid}[t]   &= \text{EMA}(\text{close},\ \text{window\_size}) \\
\text{atr}[t]   &= \text{ATR}(\text{high},\ \text{low},\ \text{close},\ \text{window\_size}) \\
\text{upper}[t] &= \text{mid} + \text{num\_atr}\ \cdot\ \text{atr} \\
\text{lower}[t] &= \text{mid} - \text{num\_atr}\ \cdot\ \text{atr}
\end{aligned}
$$

**3-input, 3-output** (`FunctorBase<_, 3, 3>`). Inputs `(high, low, close)`; outputs
`(lower, mid, upper)`. First valid at sample index `window_size`.

Composes one `EwMean(span=window_size)` for the midline + one `ATR(window_size)`. O(1)
per step.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->
