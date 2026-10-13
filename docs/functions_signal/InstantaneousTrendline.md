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
- adaptive moving average
- trendline
short: Ehlers' instantaneous trendline, the mean over one measured dominant cycle.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- DominantCycle
- MAMA
- SuperSmoother
- KAMA
references:
- 'Ehlers, J. F. (2001). *Rocket Science for Traders: Digital Signal Processing Applications*.
  Wiley.'
---

# `InstantaneousTrendline`

## Description

`InstantaneousTrendline` is John Ehlers' instantaneous trendline (*Rocket
Science for Traders*, "The Instantaneous Trendline"), as TA-Lib implements it
in `HT_TRENDLINE`. It averages the input over one dominant cycle, measured at
every sample by the same Hilbert-transform engine as `DominantCycle`, then
smooths the average with a 4-tap weighted mean. An average over exactly one
cycle removes that cycle, so the trendline keeps the trend and drops the
dominant cycle.

$$
\begin{aligned}
N[t] &= \lfloor \text{period}[t] + 0.5 \rfloor \\
\text{itrend}[t] &= \frac{1}{N[t]} \sum_{k=0}^{N[t]-1} x[t-k] \\
\text{trendline}[t] &= \frac{4\,\text{itrend}[t] + 3\,\text{itrend}[t-1]
  + 2\,\text{itrend}[t-2] + \text{itrend}[t-3]}{10}
\end{aligned}
$$

`period` is the dominant cycle period in samples that `DominantCycle`
reports, limited to `[6, 50]`. The average costs lag: on a steady ramp the
trendline trails the price by `(N - 1) / 2 + 1` samples, half a cycle plus
one. A trend also biases the period estimate (see `DominantCycle`), and an
average over a cycle length that does not match the actual cycle leaves part
of the cycle in the output.

This is a parameterless function: the averaging window comes entirely from the
measured cycle.

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
