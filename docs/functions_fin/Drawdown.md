---
name: Drawdown
title: Drawdown
implementation_family: fin
topics:
- risk
tags:
- drawdown
short: Running drawdown from the cumulative peak.
inputs: 1
outputs: 1
parameters: []
---

# `Drawdown`

## Description

Running drawdown from the cumulative-since-inception peak:

$$
\text{Drawdown}[t] = \frac{\text{price}[t]}{\text{CumMax}(\text{price})[t]} - 1
$$

A flat or new-high series gives `0`. A 30 % loss from the prior peak gives `-0.30`.
Composes `CumMax`. No warmup.

## Notes

- Bit-exact to a `pandas.Series.cummax`-based reference.
- See also `MaxDrawdown` (running min of `Drawdown`) and `RollingMaxDrawdown` (worst drawdown inside a trailing window).

<!-- HELP_END -->
