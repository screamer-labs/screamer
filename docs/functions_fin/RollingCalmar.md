---
name: RollingCalmar
title: Rolling Calmar ratio
implementation_family: fin
topics:
- risk
tags:
- calmar
- ratio
- rolling
- drawdown
short: Annualised return divided by the worst rolling drawdown.
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
---

# `RollingCalmar`

## Description

Calmar ratio: annualised return divided by the worst rolling drawdown:

$$
\text{Calmar}[t] = \frac{\text{ppy}\ \cdot\ \text{mean}(r)}{\big|\,\text{RollingMaxDrawdown}(\text{implied price})\,\big|}
$$

Takes a *returns* series; internally reconstructs the implied price path as a cumulative
product `price *= (1 + r)` starting from 1.0, so the drawdown calculation is well-defined.
Returns `NaN` when the path is monotonic up (no drawdown in window).

If you already have a price series, compose by hand:

```python
calmar = ppy * RollingMean(window)(returns) / abs(RollingMaxDrawdown(window)(price))
```

<!-- HELP_END -->
