---
name: CyclePhase
title: Instantaneous cycle phase
implementation_family: signal
topics:
- cycles
tags:
- ehlers
- hilbert
- cycle
- analytic-signal
short: Instantaneous phase, in degrees, of the analytic signal via Ehlers' homodyne discriminator.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `CyclePhase`

## Description

`CyclePhase` reads the instantaneous phase of the analytic signal formed from
a series by John Ehlers' Hilbert transform, in degrees, wrapped to `[0, 360)`.
It shares the same in-phase and quadrature components as `HilbertPhasor` and
the same period estimate as `DominantCycle`: the phase is the angle of the
point `(I, Q)` in the complex plane, and it advances by roughly `360 /
period` degrees per sample while the dominant cycle stays stable. The
estimate is a general spectral quantity: the same measurement applies to any
sampled signal, not only price.

*Parameters*: `CyclePhase` takes no parameters.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->
