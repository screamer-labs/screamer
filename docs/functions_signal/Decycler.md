---
name: Decycler
title: Ehlers Decycler trend filter
implementation_family: signal
topics:
- smoothing
- filtering
tags:
- ehlers
- iir
- trend
short: Ehlers trend estimate with short cycles removed.
inputs: 1
outputs: 1
parameters:
- name: period
  type: float
  default: 60.0
  description: Cycles at or below this length in samples are removed. Give this or cutoff.
- name: cutoff
  type: float
  default: null
  description: Cutoff as a fraction of Nyquist in (0, 1). Give this or period, not both.
nan_policy: ignore
---

# `Decycler`

## Description

`Decycler` is a causal filter from John Ehlers that estimates the trend of a
series by removing cycles at or below the cutoff. It is the input minus its
1-pole highpass component, so slow movements pass through and fast cycles are
suppressed.

The cutoff is set by a cycle length in samples (`period`) or a fraction of the
Nyquist frequency (`cutoff`), related by `cutoff = 2 / period`.

### Parameters

**`period`** *(float)*: The shortest cycle length in samples that is removed.

**`cutoff`** *(float)*: The cutoff as a fraction of the Nyquist frequency, in (0, 1). Provide exactly one of `period` or `cutoff`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
