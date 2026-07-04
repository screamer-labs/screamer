---
name: MFI
title: Money Flow Index (MFI)
implementation_family: rolling
topics:
- volume
- oscillator
tags:
- mfi
- money-flow
- volume-rsi
- talib
- ohlcv
short: Volume-weighted analogue of RSI on the typical price.
inputs: 4
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Lookback period (Wilder's default is 14).
nan_policy: ignore
---

# `MFI`

## Description

Money Flow Index - a volume-weighted analogue of RSI on the typical price:

$$
\begin{aligned}
\text{TP}[t]     &= (H + L + C) / 3 \\
\text{MF}[t]     &= \text{TP} \cdot V \\
\text{pos\_MF}_w &= \sum_w \text{MF}[\text{where}\ \text{TP} > \text{TP}_{t-1}] \\
\text{neg\_MF}_w &= \sum_w \text{MF}[\text{where}\ \text{TP} < \text{TP}_{t-1}] \\
\text{MFI}[t]    &= 100\ \cdot\ \dfrac{\text{pos\_MF}_w}{\text{pos\_MF}_w + \text{neg\_MF}_w}
\end{aligned}
$$

**4-input, 1-output** on `(high, low, close, volume)`. First valid output at sample index
`window_size`. Bit-exact to `talib.MFI` (~1e-14).


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->
