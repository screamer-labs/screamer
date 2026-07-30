---
name: RoofingFilter
title: Ehlers roofing filter (bandpass)
implementation_family: signal
topics:
- filtering
tags:
- ehlers
- iir
- bandpass
short: Ehlers bandpass, a highpass then a SuperSmoother.
inputs: 1
outputs: 1
parameters:
- name: hp_period
  type: float
  default: 48.0
  description: Highpass cutoff as a cycle length in samples. Give this or hp_cutoff.
- name: lp_period
  type: float
  default: 10.0
  description: Lowpass cutoff as a cycle length in samples. Give this or lp_cutoff.
- name: hp_cutoff
  type: float
  default: null
  description: Highpass cutoff as a fraction of Nyquist in (0, 1). Give this or hp_period.
- name: lp_cutoff
  type: float
  default: null
  description: Lowpass cutoff as a fraction of Nyquist in (0, 1). Give this or lp_period.
nan_policy: ignore
---

# `RoofingFilter`

## Description

`RoofingFilter` is a causal bandpass from John Ehlers. It applies a 2-pole
highpass at `hp_period` to remove the trend, then a `SuperSmoother` at
`lp_period` to remove fast noise, leaving the band of cycles between the two
cutoffs. It is the preprocessing step Ehlers applies before measuring a cycle,
because a raw series carries a trend and high-frequency noise that distort the
measurement.

Each cutoff is a cycle length in samples (`hp_period`, `lp_period`) or a fraction
of the Nyquist frequency (`hp_cutoff`, `lp_cutoff`), related by `cutoff = 2 / period`.

### Parameters

**`hp_period`** / **`hp_cutoff`** *(float)*: The highpass cutoff. Provide exactly one.

**`lp_period`** / **`lp_cutoff`** *(float)*: The lowpass cutoff. Provide exactly one.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
