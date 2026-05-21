---
name: RollingResidualStd
title: Rolling residual std
implementation_family: fin
topics:
- correlation
- regression
tags:
- residual
- spread
- std
- pair
- pairs-trading
short: Standard deviation of the rolling-hedge-adjusted residual y - beta*x.
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

# `RollingResidualStd`

## Description

Standard deviation of the per-bar hedge-adjusted spread `y − β·x` over the trailing window
(sample std, ddof=1):

$$
\sigma_\text{spread}[t] = \text{RollingStd}\big(\text{RollingSpread}(y, x)\big)[t]
$$

Useful for pairs-trading z-score normalisation:


Composes `RollingSpread` + `RollingStd`. O(1) per step. NaN-poisoning during
`RollingSpread`'s own warmup is gated explicitly so the std accumulator stays clean.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Description

```python
z = (current_spread - mean_spread) / RollingResidualStd(60)(y, x)
```

<!-- HELP_END -->
