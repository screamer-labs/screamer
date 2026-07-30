---
name: DominantCycle
title: Dominant cycle period
implementation_family: signal
topics:
- cycles
tags:
- ehlers
- hilbert
- cycle
short: Dominant cycle period via Ehlers' homodyne discriminator.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `DominantCycle`

## Description

`DominantCycle` measures the dominant cycle period of a series, in samples,
using John Ehlers' homodyne discriminator. It forms the analytic signal by a
Hilbert transform, then reads the cycle period from the rate of change of the
phase between successive samples. The estimate is a general spectral quantity:
the same measurement applies to any sampled signal, not only price.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
