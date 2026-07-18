---
name: RollingCVaR
title: Rolling Conditional Value-at-Risk (Expected Shortfall)
implementation_family: fin
topics:
- risk
tags:
- cvar
- conditional value at risk
- expected shortfall
- tail risk
- var
- risk
short: "Historical Conditional Value-at-Risk (Expected Shortfall): the mean loss in the worst alpha tail over a window."
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Window length in observations.
- name: alpha
  type: float
  default: 0.05
  description: Tail probability (e.g. 0.05 for the worst 5%).
nan_policy: ignore
see_also:
- RollingQuantile
- RollingSortino
---

# `RollingCVaR`

## Description

`RollingCVaR` is the historical Conditional Value-at-Risk, or Expected Shortfall,
of a return series over a trailing window. With the worst
`k = max(1, floor(alpha * window))` returns in the window,

    CVaR = -mean(the k smallest returns),

a positive number: the average loss you suffer in the worst `alpha` tail. Unlike
plain Value-at-Risk, CVaR is a coherent tail-risk measure. Value-at-Risk is only
the tail quantile itself, `-RollingQuantile(window, alpha)`; CVaR averages the
losses *beyond* it, so it captures how bad the tail gets, not just where it starts.
The window is held in an order-statistic tree (`O(log W)` to update, `O(k)` to read
the tail). The output is `NaN` until the window is full; a `NaN` return leaves the
window untouched (`nan_policy: ignore`).

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingCVaR, RollingQuantile

    rng = np.random.default_rng(2)
    n = 500
    r = rng.standard_normal(n) * 0.01
    alpha = 0.05
    cvar = RollingCVaR(100, alpha=alpha)(r)
    var = -RollingQuantile(100, alpha)(r)               # VaR is just the negated tail quantile

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=r, name='returns', line=dict(color='lightslategray'), opacity=0.6),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=var, name='VaR (5%)', line=dict(color='darkorange')), row=2, col=1)
    fig.add_trace(go.Scatter(y=cvar, name='CVaR / expected shortfall (5%)',
                             line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(title='RollingCVaR: the average loss beyond the VaR tail',
                      yaxis=dict(title='return'), yaxis2=dict(title='loss'),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
