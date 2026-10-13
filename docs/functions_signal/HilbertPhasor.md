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
short: Ehlers' in-phase and quadrature components (TA-Lib HT_PHASOR), scaled to the input amplitude.
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

`HilbertPhasor` returns the in-phase and quadrature components `I1` and `Q1`
of John Ehlers' Hilbert transform (*Rocket Science for Traders*), the pair
TA-Lib returns from `HT_PHASOR`, from the same engine as `DominantCycle`. `I1`
is the detrended, smoothed input delayed by three samples, and `Q1` is its
Hilbert transform. `Q1` leads `I1` by a quarter cycle, Ehlers' convention.
Together they describe the instantaneous phase (`CyclePhase`) and amplitude
(`CycleAmplitude`) of the dominant cycle. The estimate is a general spectral
quantity: the same measurement applies to any sampled signal, not only price.

Scale: Ehlers corrects the Hilbert filter's gain with the factor
`0.075 * period + 0.54`, which only approximates unit gain, and TA-Lib returns
the components with that approximate scale (a unit sine reads between about
0.55 and 1.04 depending on its period). Neither defines an exact amplitude, so
screamer divides each component by its filter gain at the measured period:
a sine of amplitude `A` and any period in `[6, 50]` gives components of
amplitude `A`. The phase relation and the shape match TA-Lib; the values
differ from `HT_PHASOR` by that per-period factor. With `w = 2 pi / period`
(the engine's period before its final smoothing) and `adj = 0.075 *
period[t-1] + 0.54`, the gains are

$$
\begin{aligned}
W &= \left| 0.4 + 0.3 e^{-iw} + 0.2 e^{-2iw} + 0.1 e^{-3iw} \right| \\
h &= 2\,\text{adj}\,(0.0962 \sin 3w + 0.5769 \sin w) \\
g_I &= W h, \qquad g_Q = W h^2
\end{aligned}
$$

and `inphase = I1 / g_I`, `quadrature = Q1 / g_Q`: `W` is the gain of the
4-tap price smoothing and `h` that of one pass of the Hilbert filter.

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
