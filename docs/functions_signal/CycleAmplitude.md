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
- envelope
- hilbert transform
- analytic signal
short: Instantaneous amplitude (envelope) of the analytic signal via Ehlers' homodyne discriminator.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- HilbertPhasor
- CyclePhase
- DominantCycle
references:
- 'Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*.
  Wiley.'
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

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import CycleAmplitude

    rng = np.random.default_rng(20)
    n = 400
    t = np.arange(n)
    envelope = 1.0 + 0.6 * np.sin(2 * np.pi * t / 220)         # slowly varying amplitude
    x = envelope * np.sin(2 * np.pi * t / 20) + 0.05 * rng.standard_normal(n)

    amp = CycleAmplitude()(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x, name='amplitude-modulated cycle', line=dict(color='lightgray')))
    fig.add_trace(go.Scatter(y=envelope, name='true envelope', line=dict(color='black', dash='dash')))
    fig.add_trace(go.Scatter(y=amp, name='CycleAmplitude()', line=dict(color='steelblue')))
    fig.update_layout(title='CycleAmplitude: envelope of an amplitude-modulated cycle',
                      xaxis_title='sample', yaxis_title='value',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
