---
name: RollingHitRate
title: Rolling hit rate
implementation_family: fin
topics:
- risk
tags:
- hit-rate
- win-rate
- rolling
short: Fraction of strictly-positive samples in a trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 252
  min: 2
  description: Trailing-window length.
---

# `RollingHitRate`

## Description

Fraction of strictly-positive samples in the trailing window:

$$
\text{HitRate}[t] = \frac{1}{w}\ \text{count}(r_i > 0,\ i \in \text{window})
$$

Output in `[0, 1]`. Composes `detail::RollingSum` over the indicator `(r > 0)`.

<!-- HELP_END -->
