---
name: RollingSortino
title: Rolling Sortino ratio
implementation_family: fin
topics:
- risk
tags:
- sortino
- ratio
- rolling
- downside
short: 'Annualised Sortino ratio: Sharpe with downside-only deviation.'
inputs: 1
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
- name: target
  type: float
  default: 0.0
  description: Minimum acceptable return (only deviations below this contribute to
    the denominator).
---

# `RollingSortino`

## Description

Annualised Sortino ratio:

$$
\text{Sortino}[t] = \sqrt{\text{ppy}}\ \cdot\ \frac{\text{mean}(r) - \text{target}}{\sqrt{\text{mean}(\min(r - \text{target},\ 0)^2)}}
$$

Same as Sharpe but the denominator is the *downside* deviation -- only bars below `target`
contribute, so upside variability is not penalised.

## Implementation

`O(window_size)` per step. The downside-RMS denominator does not have a closed-form
O(1) update.

<!-- HELP_END -->
