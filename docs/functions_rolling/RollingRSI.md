---
name: RollingRSI
title: Rolling RSI
implementation_family: rolling
topics:
- oscillator
tags:
- rsi
- wilder
- cutler
- momentum
- oscillator
short: Relative Strength Index. Wilder's smoothing by default; Cutler's via method='cutler'.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Period. Wilder's original choice is 14.
- name: method
  type: str
  default: wilder
  enum:
  - wilder
  - cutler
  description: Smoothing convention. 'wilder' matches TA-Lib and pandas-ta; 'cutler'
    uses a plain SMA.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behaviour.
---

# `RollingRSI`

## Description

Wilder's Relative Strength Index. Up- and down-moves are separately smoothed; the index
is `100 - 100 / (1 + RS)` where `RS` is the smoothed ratio of average up to average down
moves:

$$
\begin{aligned}
\Delta[t] &= x[t] - x[t-1] \\
U[t] &= \max(\Delta, 0),\quad D[t] = \max(-\Delta, 0) \\
\overline{U}[t] &= \text{smooth}(U,\ \text{window\_size}),\quad \overline{D}[t] = \text{smooth}(D,\ \text{window\_size}) \\
\text{RS}[t] &= \overline{U}[t] / \overline{D}[t] \\
\text{RSI}[t] &= 100 - 100 / (1 + \text{RS})
\end{aligned}
$$

The smoothing depends on `method`:

- `'wilder'` (default) -- Wilder's smoothing
  $\overline{X}[t] = \overline{X}[t-1] + (X[t] - \overline{X}[t-1]) / w$, with the
  SMA-of-the-first-`w` seed. Matches `talib.RSI` and `pandas-ta-classic.rsi` bit-exactly
  post-warmup.
- `'cutler'` -- plain rolling-mean smoothing. Cutler's variant. Useful if you specifically
  want the SMA-form (some legacy systems use it). Does **not** match TA-Lib.

## Notes

- First valid output at sample index `window_size`.
- Output is bounded in `[0, 100]`.
- The default was changed from 'cutler' to 'wilder' in a recent release to align with
  TA-Lib and pandas-ta-classic. Pass `method='cutler'` to recover the old behaviour.

<!-- HELP_END -->
