---
name: CycleAmplitude
title: Instantaneous cycle amplitude
implementation_family: signal
topics:
- cycles
tags:
- ehlers
- hilbert
- cycle
- analytic-signal
short: Instantaneous amplitude (envelope) of the analytic signal via Ehlers' homodyne discriminator.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `CycleAmplitude`

## Description

`CycleAmplitude` reads the instantaneous amplitude, or envelope, of the
analytic signal formed from a series by John Ehlers' Hilbert transform. It
shares the same in-phase and quadrature components as `HilbertPhasor` and
`CyclePhase`: the amplitude is the magnitude of the point `(I, Q)` in the
complex plane, `sqrt(I^2 + Q^2)`. It is a general envelope detector, tracking
the local size of any oscillation in the series regardless of its period, and
applies to any sampled signal, not only price.

*Parameters*: `CycleAmplitude` takes no parameters.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
