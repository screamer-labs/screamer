---
name: RollingDownsideDeviation
title: Rolling Downside Deviation
implementation_family: fin
topics:
- risk
- volatility
tags:
- downside deviation
- downside risk
- semideviation
- sortino
- risk
short: "Trailing-window downside semideviation: the RMS of returns falling below a minimum acceptable return."
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Window length in observations.
- name: mar
  type: float
  default: 0.0
  description: Minimum acceptable return; only returns below it contribute.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behaviour.
nan_policy: ignore
see_also:
- RollingSortino
- RollingStd
---

# `RollingDownsideDeviation`

## Description

`RollingDownsideDeviation` is the downside semideviation over a trailing window:
the root-mean-square of the shortfalls below a minimum acceptable return `mar`,

    sqrt( mean( min(return - mar, 0)^2 ) ).

Only returns below `mar` contribute; anything at or above it counts as zero. It is
the one-sided risk measure that penalizes losses but not upside volatility, and it
is exactly the denominator of the Sortino ratio, exposed on its own. A `NaN`
return leaves the window untouched and yields `NaN` (`nan_policy: ignore`).

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingDownsideDeviation, RollingStd

    rng = np.random.default_rng(0)
    n = 500
    # returns with occasional sharp losses (left skew), so downside risk exceeds total vol
    r = rng.standard_normal(n) * 0.01 - (rng.random(n) < 0.05) * 0.03
    dd = RollingDownsideDeviation(60)(r)
    sd = RollingStd(60)(r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=sd, name='total std', line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(y=dd, name='downside deviation', line=dict(color='crimson')))
    fig.update_layout(title='RollingDownsideDeviation: only the shortfalls count',
                      xaxis_title='bar', yaxis_title='per-bar risk',
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
