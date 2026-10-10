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
- sinewave indicator
- hilbert transform
- analytic signal
short: Sinewave indicator (sine, leadsine) from the instantaneous phase of the analytic signal.
inputs: 1
outputs: 2
parameters: []
nan_policy: ignore
see_also:
- CyclePhase
- HilbertPhasor
- TrendMode
references:
- 'Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*.
  Wiley.'
output_names:
- sine
- leadsine
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

The two outputs are stacked along a trailing axis of size 2: `out[..., 0]`
is `sine` and `out[..., 1]` is `leadsine`.

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
    from screamer import CycleSine

    rng = np.random.default_rng(5)
    n = 300
    true_period = np.where(np.arange(n) < n // 2, 20.0, 30.0)  # period changes halfway
    x = np.sin(np.cumsum(2 * np.pi / true_period)) + 0.05 * rng.standard_normal(n)

    out = CycleSine()(x)
    sine, leadsine = out[:, 0], out[:, 1]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=x, name='input', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=sine, name='sine', line=dict(color='steelblue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=leadsine, name='leadsine', line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(title='CycleSine: sine and leadsine cross near the turning points of the cycle',
                      yaxis=dict(title='input'), yaxis2=dict(title='indicator'),
                      xaxis2=dict(title='sample'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
