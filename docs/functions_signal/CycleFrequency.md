---
name: CycleFrequency
title: Instantaneous cycle frequency
implementation_family: signal
topics:
- cycles
tags:
- ehlers
- hilbert
- cycle
short: Instantaneous frequency, in cycles per sample, via Ehlers' homodyne discriminator.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `CycleFrequency`

## Description

`CycleFrequency` measures the instantaneous frequency of a series, in cycles
per sample, using John Ehlers' homodyne discriminator. It is the reciprocal
of the dominant cycle period reported by `DominantCycle`: a period of 20
samples corresponds to a frequency of 0.05 cycles per sample. The estimate is
a general spectral quantity: the same measurement applies to any sampled
signal, not only price.

*Parameters*: `CycleFrequency` takes no parameters.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
