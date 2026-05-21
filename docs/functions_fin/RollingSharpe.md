---
name: RollingSharpe
title: Rolling Sharpe ratio
implementation_family: fin
topics:
- risk
tags:
- sharpe
- ratio
- rolling
short: Annualised Sharpe ratio over a trailing window of returns.
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
nan_policy: ignore
---

# `RollingSharpe`

## Description

Annualised Sharpe ratio over a trailing window of returns:

$$
\text{Sharpe}[t] = \sqrt{\text{ppy}}\ \cdot\ \frac{\text{RollingMean}(r)}{\text{RollingStd}(r)}
$$

Composes `RollingMean` + `RollingStd` (sample std, ddof=1 to match pandas). Returns `NaN`
where the std is zero.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->
