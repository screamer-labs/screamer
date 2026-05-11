---
name: DonchianChannels
title: Donchian Channels
implementation_family: rolling
topics:
- channels
tags:
- donchian
- channels
- breakout
- envelope
short: 'Trend-following envelope: rolling max(high), rolling min(low), and midline.'
inputs: 2
outputs: 3
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Window for the rolling max/min.
---

# `DonchianChannels`

## Description

Trend-following envelope. The upper line is the rolling max of `high`, the lower line is
the rolling min of `low`, and the midline is their average:

$$
\begin{aligned}
\text{upper}[t] &= \max(\text{high},\ w\ \text{bars}) \\
\text{lower}[t] &= \min(\text{low},\ w\ \text{bars}) \\
\text{mid}[t]   &= (\text{upper} + \text{lower}) / 2
\end{aligned}
$$

**2-input, 3-output** (`FunctorBase<_, 2, 3>`). Inputs `(high, low)`; outputs
`(lower, mid, upper)`. First valid at sample index `window_size - 1`.

Composes two `detail::MonotonicDeque` instances. Amortised O(1) per step. Bit-exact to
`pandas-ta-classic.donchian`.

<!-- HELP_END -->
