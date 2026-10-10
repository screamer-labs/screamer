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
- cycle period
- hilbert transform
- mesa
short: Dominant cycle period via Ehlers' homodyne discriminator.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- CycleFrequency
- CyclePhase
- HilbertPhasor
- InstantaneousTrendline
references:
- 'Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*.
  Wiley.'
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
    from screamer import DominantCycle

    rng = np.random.default_rng(20)
    n = 600
    true_period = np.where(np.arange(n) < n // 2, 16.0, 30.0)  # period steps from 16 to 30
    x = np.sin(np.cumsum(2 * np.pi / true_period)) + 0.08 * rng.standard_normal(n)

    period = DominantCycle()(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=x, name='input', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=true_period, name='true period', line=dict(color='black', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(y=period, name='DominantCycle()', line=dict(color='steelblue')), row=2, col=1)
    fig.update_layout(title='DominantCycle: measured period follows a step from 16 to 30 samples',
                      yaxis=dict(title='input'), yaxis2=dict(title='period (samples)', range=[0, 40]),
                      xaxis2=dict(title='sample'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
