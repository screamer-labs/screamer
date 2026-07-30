---
name: TrendMode
title: Trend-vs-cycle classifier
implementation_family: signal
topics:
- cycles
tags:
- ehlers
- hilbert
- cycle
- analytic-signal
short: Classifies each sample as trending (1.0) or cycling (0.0) from the dominant-cycle phase advance.
inputs: 1
outputs: 1
parameters:
- name: phase_rate_frac
  type: float
  default: 0.5
  min: 0.0
  description: Threshold, as a fraction of the expected per-sample phase advance for the current dominant cycle, below which a sample is classified as trending.
nan_policy: ignore
---

# `TrendMode`

## Description

`TrendMode` classifies each sample of a series as trending or cycling, using
the same instantaneous phase and dominant-cycle period as `CyclePhase` and
`DominantCycle`. While a series oscillates at a stable dominant cycle, the
phase advances by close to `360 / period` degrees per sample. When the phase
advance stalls well below that rate, the dominant cycle has effectively
stopped rotating, which this function reads as a trending market.

The rule: at each sample, compute the phase change from the previous sample
(unwrapped to `[-180, 180]` degrees) and compare its magnitude to
`phase_rate_frac * (360 / period)`, the expected per-sample advance scaled
by `phase_rate_frac`. If the phase change is smaller, the output is `1.0`
(trending); otherwise it is `0.0` (cycling).

This is a threshold classifier, not a transcription of any single reference
formula: it is one simple, causal way to turn the phase and period estimates
into a binary trend/cycle flag. Lowering `phase_rate_frac` makes the
classifier stricter about calling a trend, raising it makes trend mode more
frequent.

*Parameters*: `TrendMode` takes one parameter.

### Parameters

**`phase_rate_frac`** *(float, default `0.5`)*: The threshold, as a fraction
of the expected per-sample phase advance for the current dominant cycle,
below which a sample is classified as trending.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
