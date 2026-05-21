---
name: RollingAlpha
title: Rolling alpha (regression intercept)
implementation_family: fin
topics:
- correlation
- regression
tags:
- alpha
- regression
- intercept
- pair
short: Rolling OLS intercept of target on regressor (companion to RollingBeta).
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behaviour.
nan_policy: ignore
---

# `RollingAlpha`

## Description

Companion to `RollingBeta`. The regression intercept of `y` on `x`:

$$
\alpha[t] = \overline{y}_w - \beta[t]\ \cdot\ \overline{x}_w
$$

**2-input, 1-output** on `(target, regressor)` -- same convention as `RollingBeta`.
Composes `RollingBeta` + two `RollingMean` instances. O(1) per step.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->
