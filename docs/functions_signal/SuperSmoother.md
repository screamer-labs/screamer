---
name: SuperSmoother
title: Ehlers SuperSmoother lowpass filter
implementation_family: signal
topics:
- smoothing
- filtering
tags:
- ehlers
- iir
- lowpass
short: Ehlers 2-pole low-lag lowpass filter.
inputs: 1
outputs: 1
parameters:
- name: period
  type: float
  default: 20.0
  description: Cutoff as a cycle length in samples. Give this or cutoff, not both.
- name: cutoff
  type: float
  default: null
  description: Cutoff as a fraction of Nyquist in (0, 1). Give this or period, not both.
nan_policy: ignore
---

# `SuperSmoother`

## Description

`SuperSmoother` is a causal 2-pole IIR lowpass filter from John Ehlers. It passes
components of the input series slower than the cutoff and attenuates faster ones,
with less lag and less ringing than a Butterworth filter of comparable rolloff.

The cutoff is set by a cycle length in samples (`period`) or, equivalently, a
fraction of the Nyquist frequency (`cutoff`), related by `cutoff = 2 / period`.
The filter coefficients follow Ehlers' critically damped design.

### Parameters

**`period`** *(float)*: The cutoff expressed as a cycle length in samples.

**`cutoff`** *(float)*: The cutoff expressed as a fraction of the Nyquist frequency, in (0, 1). Provide exactly one of `period` or `cutoff`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
