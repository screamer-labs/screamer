---
name: MAMA
title: MESA Adaptive Moving Average (MAMA)
implementation_family: signal
topics:
- smoothing
tags:
- ehlers
- adaptive
- mesa
- adaptive moving average
- fama
short: Ehlers' adaptive moving average pair (MAMA, FAMA), smoothing set by the rate of the cycle
  phase.
inputs: 1
outputs: 2
parameters:
- name: fast_limit
  type: float
  default: 0.5
  description: Upper bound on the smoothing constant, reached when the phase stalls or wraps.
- name: slow_limit
  type: float
  default: 0.05
  description: Lower bound on the smoothing constant, reached when the phase changes by fast_limit
    / slow_limit degrees or more per sample.
nan_policy: ignore
see_also:
- KAMA
- InstantaneousTrendline
- CyclePhase
- EwMean
references:
- 'Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*.
  Wiley.'
output_names:
- mama
- fama
---

# `MAMA`

## Description

`MAMA` (MESA Adaptive Moving Average, John Ehlers, *Stocks & Commodities*,
September 2001, and *Rocket Science for Traders*) is an exponential moving
average whose smoothing constant changes every sample with the rate of the
cycle phase, as TA-Lib implements it in `MAMA`. The constant is `fast_limit`
divided by the phase change per sample. While a cycle of period `P` rotates
steadily the phase changes by about `360 / P` degrees per sample, so the
constant is small, about `fast_limit * P / 360`, and `MAMA` smooths heavily.
Where the phase stalls or wraps, the change drops to the 1-degree floor and
the constant jumps to `fast_limit`, so `MAMA` moves most of the way to the
price in one step. `FAMA` (following adaptive moving average) is a second pass
over `MAMA` at half the smoothing constant, and lags it.

$$
\begin{aligned}
\Delta\phi[t]   &= \max\big(\phi[t-1] - \phi[t],\ 1\big) \\
\alpha[t]       &= \operatorname{clip}\!\left(\frac{\text{fast\_limit}}{\Delta\phi[t]},\ \text{slow\_limit},\ \text{fast\_limit}\right) \\
\text{MAMA}[t]  &= \alpha[t] \cdot x[t] + (1 - \alpha[t]) \cdot \text{MAMA}[t-1] \\
\text{FAMA}[t]  &= 0.5\,\alpha[t] \cdot \text{MAMA}[t] + (1 - 0.5\,\alpha[t]) \cdot \text{FAMA}[t-1]
\end{aligned}
$$

$\phi$ is Ehlers' phase $\arctan(Q_1 / I_1)$ in degrees, from the in-phase and
quadrature components of the Hilbert transform (`HilbertPhasor`), taken as 0
when $I_1 = 0$. It lies in $(-90, 90)$, so it wraps every half cycle and
`alpha` steps to `fast_limit` twice per cycle. The phase change $\Delta\phi$
is floored at 1 degree so `alpha` never exceeds `fast_limit`. `MAMA` and
`FAMA` are both seeded with the first input after the warm-up.

The two outputs are stacked along a trailing axis of size 2: `out[..., 0]`
is `MAMA` and `out[..., 1]` is `FAMA`.

The results agree with TA-Lib's `MAMA` once its start-up has faded (TA-Lib
starts both averages at 0).

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
