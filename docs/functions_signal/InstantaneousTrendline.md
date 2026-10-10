---
name: InstantaneousTrendline
title: Instantaneous trendline
implementation_family: signal
topics:
- smoothing
tags:
- ehlers
- adaptive
- trend
short: Adaptive 2-pole trendline whose smoothing tracks the measured dominant cycle.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `InstantaneousTrendline`

## Description

`InstantaneousTrendline` (John Ehlers) is an adaptive trendline: a 2-pole
recursive filter whose smoothing factor is set, at every sample, from the
dominant cycle period measured by the same Hilbert-transform engine as
`DominantCycle`. Because the filter's time constant tracks the current cycle
length, the trendline removes the dominant cycle from the output and follows
the underlying trend rather than a fixed-period moving average.

$$
\begin{aligned}
\alpha[t] &= \frac{2}{\text{period}[t] + 1} \\
\text{it}[t] &= \left(\alpha[t] - \frac{\alpha[t]^2}{4}\right) x[t]
  + \frac{\alpha[t]^2}{2} x[t-1]
  - \left(\alpha[t] - \frac{3\alpha[t]^2}{4}\right) x[t-2] \\
  &\quad + 2(1 - \alpha[t]) \, \text{it}[t-1] - (1 - \alpha[t])^2 \, \text{it}[t-2]
\end{aligned}
$$

`period` is the dominant cycle period in samples, the same quantity
`DominantCycle` reports; `HilbertCycle` clamps it to `[6, 50]`, which bounds
`alpha` to at most `2/7`. The recursion needs two samples of input and output
history; until that history is available (right after the dominant cycle
first becomes measurable), `InstantaneousTrendline` seeds its output with the
current price.

This is a parameterless function: the adaptation comes entirely from the
measured cycle, with no smoothing constant to tune.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Reference

Ehlers, *Cybernetic Analysis for Stocks and Futures* (2004). There is no
bit-exact reference implementation for this recursion; correctness is gated
by causality (batch and streaming agree) and by tracking behavior on a clean
trend.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import InstantaneousTrendline

    rng = np.random.default_rng(9)
    n = 500
    t = np.arange(n)
    trend = 100 + 4.0 * np.sin(2 * np.pi * t / 450)             # slow trend
    period = np.linspace(15, 35, n)                             # cycle period drifts from 15 to 35
    x = trend + 1.0 * np.sin(np.cumsum(2 * np.pi / period)) + 0.2 * rng.standard_normal(n)

    itrend = InstantaneousTrendline()(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x, name='trend + drifting cycle + noise', line=dict(color='lightgray')))
    fig.add_trace(go.Scatter(y=trend, name='underlying trend', line=dict(color='black', dash='dash')))
    fig.add_trace(go.Scatter(y=itrend, name='InstantaneousTrendline()', line=dict(color='steelblue')))
    fig.update_layout(title='InstantaneousTrendline: trend of a series whose cycle period drifts',
                      xaxis_title='sample', yaxis_title='value',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
