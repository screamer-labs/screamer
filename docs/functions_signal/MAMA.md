---
name: MAMA
title: MESA Adaptive Moving Average (MAMA)
implementation_family: signal
topics:
- smoothing
tags:
- ehlers
- adaptive
short: Adaptive moving average pair (MAMA, FAMA) whose smoothing rate tracks the instantaneous-phase rate of change.
inputs: 1
outputs: 2
parameters:
- name: fast_limit
  type: float
  default: 0.5
  description: Upper bound on the smoothing constant, applied when the phase advances quickly.
- name: slow_limit
  type: float
  default: 0.05
  description: Lower bound on the smoothing constant, applied when the phase stalls.
nan_policy: ignore
---

# `MAMA`

## Description

`MAMA` (MESA Adaptive Moving Average, John Ehlers) smooths the input with a
per-step smoothing constant that adapts to the rate of change of the
instantaneous phase of the analytic signal formed from the series by the
Hilbert transform, the same phase estimate as `CyclePhase`. When the phase
advances quickly, a new cycle is underway and the smoothing constant rises
toward `fast_limit`; when the phase advance stalls, the smoothing constant
falls toward `slow_limit`. `FAMA` (following adaptive moving average) is a
second pass over `MAMA` at half the smoothing constant, and lags it.

$$
\begin{aligned}
\Delta\phi[t]   &= \max\big(\phi[t-1] - \phi[t],\ 1\big) \\
\alpha[t]       &= \operatorname{clip}\!\left(\frac{\text{fast\_limit}}{\Delta\phi[t]},\ \text{slow\_limit},\ \text{fast\_limit}\right) \\
\text{MAMA}[t]  &= \alpha[t] \cdot x[t] + (1 - \alpha[t]) \cdot \text{MAMA}[t-1] \\
\text{FAMA}[t]  &= 0.5\,\alpha[t] \cdot \text{MAMA}[t] + (1 - 0.5\,\alpha[t]) \cdot \text{FAMA}[t-1]
\end{aligned}
$$

`phi` is the instantaneous phase in degrees; `DeltaPhase` is floored at 1
degree so `alpha` never exceeds `fast_limit`. `MAMA` and `FAMA` are both
seeded with the first finite input.

This is a **1-input, 2-output** function (`FunctorBase<_, 1, 2>`). Outputs
are stacked along a trailing axis of size 2: `out[..., 0]` is `MAMA`,
`out[..., 1]` is `FAMA`.

## Parameters

- `fast_limit` (float, default `0.5`): upper bound on the smoothing constant.
- `slow_limit` (float, default `0.05`): lower bound on the smoothing constant.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Reference

Ehlers, *Rocket Science for Traders* (2001). TA-Lib's `MAMA` uses the same
adaptation rule over a different analytic-signal front end, so the two agree
in shape and turning points but not to numerical precision.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import MAMA

    rng = np.random.default_rng(8)
    n = 500
    t = np.arange(n)
    drift = np.concatenate([np.zeros(200), np.full(150, 0.08), np.full(150, -0.05)])   # flat, up, down
    x = 100 + np.cumsum(drift) + 1.5 * np.sin(2 * np.pi * t / 25) + 0.3 * rng.standard_normal(n)

    out = MAMA(fast_limit=0.5, slow_limit=0.05)(x)
    mama, fama = out[:, 0], out[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x, name='cycle + piecewise trend + noise', line=dict(color='lightgray')))
    fig.add_trace(go.Scatter(y=mama, name='MAMA', line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(y=fama, name='FAMA', line=dict(color='crimson')))
    fig.update_layout(title='MAMA and FAMA on a cycling series with changing trend',
                      xaxis_title='sample', yaxis_title='value',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
