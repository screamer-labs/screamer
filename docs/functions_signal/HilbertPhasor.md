---
name: HilbertPhasor
title: Hilbert in-phase / quadrature components
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
- in-phase
- quadrature
short: In-phase and quadrature components of the analytic signal via Ehlers' Hilbert transform.
inputs: 1
outputs: 2
parameters: []
nan_policy: ignore
see_also:
- CycleAmplitude
- CyclePhase
- DominantCycle
references:
- 'Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*.
  Wiley.'
output_names:
- inphase
- quadrature
---

# `HilbertPhasor`

## Description

`HilbertPhasor` computes the in-phase and quadrature components of the
analytic signal formed from a series by John Ehlers' Hilbert transform. The
in-phase component is the real part of the analytic signal and the
quadrature component is its imaginary part, a version of the input
phase-shifted by 90 degrees. Together they describe the instantaneous phase and
amplitude of the dominant cycle in the input, the same construction used by
`DominantCycle`. The estimate is a general spectral quantity: the same
measurement applies to any sampled signal, not only price.

The two outputs are stacked along a trailing axis of size 2: `out[..., 0]`
is the in-phase component and `out[..., 1]` the quadrature component.

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
    from screamer import HilbertPhasor

    rng = np.random.default_rng(20)
    n = 300
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 20) + 0.005 * t + 0.1 * rng.standard_normal(n)   # 20-sample cycle

    out = HilbertPhasor()(x)
    inphase, quadrature = out[:, 0], out[:, 1]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=x, name='input', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=inphase, name='in-phase', line=dict(color='steelblue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=quadrature, name='quadrature', line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(title='HilbertPhasor: in-phase and quadrature components a quarter cycle apart',
                      yaxis=dict(title='input'), yaxis2=dict(title='component'),
                      xaxis2=dict(title='sample'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
