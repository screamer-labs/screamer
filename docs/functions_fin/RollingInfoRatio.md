---
name: RollingInfoRatio
title: Rolling information ratio
implementation_family: fin
topics:
- risk
tags:
- info-ratio
- active-return
- rolling
- pair
short: 'Annualised information ratio: Sharpe of active returns against a benchmark.'
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 252
  min: 2
  description: Trailing-window length.
- name: periods_per_year
  type: float
  default: 1.0
  min: 1.0
  description: Annualisation factor (252 daily, 52 weekly, 12 monthly, 1 = no annualisation).
---

# `RollingInfoRatio`

## Description

Information ratio against a benchmark:

$$
\text{IR}[t] = \sqrt{\text{ppy}}\ \cdot\ \frac{\text{mean}(r - b)}{\text{std}(r - b)}
$$

**2-input, 1-output** on `(returns, benchmark)`. Effectively `RollingSharpe` applied to the
active-return series `r - b`.

<!-- HELP_END -->
