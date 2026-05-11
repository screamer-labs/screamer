---
name: MaxDrawdown
title: Maximum drawdown
implementation_family: fin
topics:
- risk
tags:
- drawdown
- max-drawdown
short: Worst drawdown experienced so far (since reset).
inputs: 1
outputs: 1
parameters: []
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

<!-- HELP_END -->
