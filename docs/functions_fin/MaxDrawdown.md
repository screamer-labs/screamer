---
name: MaxDrawdown
title: Maximum drawdown
implementation_family: fin
topics:
- cumulative
- risk
tags:
- drawdown
- max-drawdown
short: Worst drawdown experienced so far (since reset).
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `MaxDrawdown`

## Description

The worst (most negative) drawdown ever observed since the start (or last `reset()`):

$$
\text{MaxDrawdown}[t] = \min_{k \le t}\ \text{Drawdown}[k]
$$

Monotonically non-increasing in time. Composes `Drawdown` + `CumMin`.

## Notes

- For the trailing-window equivalent see `RollingMaxDrawdown`.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->
