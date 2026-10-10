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
- hilbert transform
- analytic signal
- instantaneous phase
short: Instantaneous phase, in degrees, of the analytic signal via Ehlers' homodyne discriminator.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- CycleSine
- HilbertPhasor
- DominantCycle
- TrendMode
references:
- 'Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*.
  Wiley.'
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
    from plotly.subplots import make_subplots
    from screamer import CyclePhase

    rng = np.random.default_rng(4)
    n = 300
    true_period = np.where(np.arange(n) < n // 2, 15.0, 30.0)  # period doubles halfway
    x = np.sin(np.cumsum(2 * np.pi / true_period)) + 0.05 * rng.standard_normal(n)

    phase = CyclePhase()(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=x, name='input', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=phase, name='CyclePhase()', line=dict(color='steelblue')), row=2, col=1)
    fig.update_layout(title='CyclePhase: phase wraps every cycle, more slowly once the period doubles',
                      yaxis=dict(title='input'),
                      yaxis2=dict(title='phase (degrees)', range=[0, 360], tickvals=[0, 90, 180, 270, 360]),
                      xaxis2=dict(title='sample'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
