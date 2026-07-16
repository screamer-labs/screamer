---
name: BulkVolumeClassifier
title: Bulk Volume Classifier (BVC)
implementation_family: micro
topics:
- microstructure
tags:
- trade sign
- bulk volume
- bvc
- order flow
- toxicity
- microstructure
short: Buy-initiated share of a bar's volume estimated as the normal CDF of return / trailing-window volatility.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Window length in observations for the trailing standard deviation.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behaviour.
nan_policy: ignore
see_also:
- TickRuleSign
- LeeReadySign
---

# `BulkVolumeClassifier`

## Description

`BulkVolumeClassifier` implements the Bulk Volume Classification (BVC) model of
Easley, Lopez de Prado, and O'Hara (2012). It estimates the buy-initiated
fraction of a bar's volume without tick-level data, using only the bar's return
and its trailing volatility.

At each step the operator computes the standardized return
`z = return_ / sigma_t`, where `sigma_t` is the rolling standard deviation of
`return_` over the most recent `window_size` observations, and evaluates the
standard normal CDF `Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))`. The result is a
fraction in `[0, 1]`: values near 1 indicate a predominantly buy-driven bar,
values near 0 indicate a sell-driven bar, and 0.5 indicates a neutral bar.

The operator maintains a rolling standard deviation of the return (two running
sums, O(1) per step) and evaluates the standard normal CDF of the standardized
return. It is causal and honors `nan_policy: ignore`; a zero-variance window
leaves the classification undefined and returns `NaN`. Batch and streaming
produce identical output.

A common pipeline is:

1. Compute bar log-returns with `LogReturn`.
2. Feed the return series to `BulkVolumeClassifier` to obtain a per-bar
   buy-fraction estimate.
3. Multiply by the bar's total volume to recover an estimated buy volume.

*Parameters*:

- **`window_size`** (`int`, >= 2): number of observations in the trailing
  standard deviation window.
- **`start_policy`** (`str`, default `"strict"`): controls the warmup period
  before `window_size` observations have been seen. `"strict"` emits `NaN`.
  `"expanding"` uses all available observations. `"zero"` fills with zero.

*Return value*: the buy fraction at each time step, in `[0, 1]`. `NaN` during
warmup (under `strict`) or when the input return is `NaN`.

**Reference**: Easley, D., Lopez de Prado, M. M., & O'Hara, M. (2012).
"Bulk classification of trading activity." *Working Paper*, Cornell University.

<!-- HELP_END -->
