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
- hilbert transform
- instantaneous frequency
short: Instantaneous frequency, in cycles per sample, via Ehlers' homodyne discriminator.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- DominantCycle
- CyclePhase
- HilbertPhasor
references:
- 'Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*.
  Wiley.'
---

# `CycleFrequency`

## Description

`CycleFrequency` measures the instantaneous frequency of a series, in cycles
per sample, using John Ehlers' homodyne discriminator. It is the reciprocal
of the dominant cycle period reported by `DominantCycle`: a period of 20
samples corresponds to a frequency of 0.05 cycles per sample. The estimate is
a general spectral quantity: the same measurement applies to any sampled
signal, not only price.

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
    from screamer import CycleFrequency

    rng = np.random.default_rng(3)
    n = 600
    true_freq = np.linspace(1 / 40, 1 / 12, n)                  # chirp: period shortens from 40 to 12
    x = np.sin(np.cumsum(2 * np.pi * true_freq)) + 0.08 * rng.standard_normal(n)

    freq = CycleFrequency()(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=x, name='chirp + noise', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=true_freq, name='true frequency', line=dict(color='black', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(y=freq, name='CycleFrequency()', line=dict(color='steelblue')), row=2, col=1)
    fig.update_layout(title='CycleFrequency: instantaneous frequency of a chirp',
                      yaxis=dict(title='input'), yaxis2=dict(title='cycles per sample', range=[0, 0.12]),
                      xaxis2=dict(title='sample'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
