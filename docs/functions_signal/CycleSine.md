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
short: Ehlers' sinewave indicator (sine, leadsine) from the phase of the dominant cycle.
inputs: 1
outputs: 2
parameters: []
nan_policy: ignore
see_also:
- CyclePhase
- DominantCycle
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

`CycleSine` computes John Ehlers' sinewave indicator (*Rocket Science for
Traders*, "The Sinewave Indicator"), as TA-Lib implements it in `HT_SINE`.
`sine` is the sine of the dominant cycle phase, and `leadsine` is the same
curve advanced by 45 degrees, one eighth of a cycle. While the series cycles,
`sine` follows the cycle in phase and `leadsine` turns first, so the two
lines cross just before the cycle's peaks and troughs. When the series trends,
the phase stops rotating and the lines run apart without crossing.

The dominant cycle phase, `DCPhase`, comes from a Fourier sum over one cycle
of the smoothed price `s` (the 4-tap weighted mean the engine uses). `period`
is the dominant cycle period that `DominantCycle` reports, and `N` is that
period rounded to whole samples:

$$
\begin{aligned}
N &= \lfloor \text{period} + 0.5 \rfloor \\
R &= \sum_{k=0}^{N-1} \sin(360 k / N)\, s[t-k], \quad
I = \sum_{k=0}^{N-1} \cos(360 k / N)\, s[t-k] \\
\text{DCPhase} &= \arctan(R / I) + 90 + 360 / \text{period}
  \;(+180 \text{ if } I < 0) \\
\text{sine} &= \sin(\text{DCPhase}), \quad
\text{leadsine} = \sin(\text{DCPhase} + 45)
\end{aligned}
$$

Angles are in degrees. The term `360 / period` compensates the one-sample
lag of the smoothing. `DCPhase` increases by about `360 / period` degrees per
sample, which is why `leadsine` leads. It is a different phase from
`CyclePhase`, which reads the angle of the `HilbertPhasor` pair and runs the
other way. The results agree with TA-Lib's `HT_SINE` once both have passed
their start-up transients.

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
