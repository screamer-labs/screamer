---
name: CycleSine
title: Cycle sinewave indicator
implementation_family: signal
topics:
- cycles
tags:
- ehlers
- hilbert
- cycle
- analytic-signal
short: Sinewave indicator (sine, leadsine) from the instantaneous phase of the analytic signal.
inputs: 1
outputs: 2
parameters: []
nan_policy: ignore
---

# `CycleSine`

## Description

`CycleSine` computes John Ehlers' sinewave indicator from the instantaneous
phase of the analytic signal formed from a series by the Hilbert transform.
It shares the same phase estimate as `CyclePhase`: given the phase in
degrees, `sine` is `sin(phase)` and `leadsine` is `sin(phase + 45deg)`, a
version of `sine` advanced by one eighth of a cycle. Because `sine` tracks
the phase directly, it turns at the same point in the cycle regardless of
the cycle's period, and the two outputs cross near the phase's turning
points. The estimate is a general spectral quantity: the same measurement
applies to any sampled signal, not only price.

This is a **1-input, 2-output** function (`FunctorBase<_, 1, 2>`). Outputs
are stacked along a trailing axis of size 2: `out[..., 0]` is `sine`,
`out[..., 1]` is `leadsine`.

*Parameters*: `CycleSine` takes no parameters.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
