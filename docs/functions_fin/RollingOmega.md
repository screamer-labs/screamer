---
name: RollingOmega
title: Rolling Omega Ratio
implementation_family: fin
topics:
- risk
tags:
- omega ratio
- keating shadwick
- performance ratio
- risk
short: "Omega ratio over a window: total gains above a threshold divided by total losses below it."
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Window length in observations.
- name: threshold
  type: float
  default: 0.0
  description: The return level separating gains from losses.
nan_policy: ignore
see_also:
- RollingSharpe
- RollingSortino
---

# `RollingOmega`

## Description

`RollingOmega` is the Omega ratio (Keating and Shadwick, 2002) over a trailing
window of returns about a `threshold`,

    omega = sum( (return - threshold)+ ) / sum( (threshold - return)+ ),

the total gain above the threshold divided by the total shortfall below it. Unlike
the Sharpe ratio, Omega uses the whole return distribution, so it reflects skew
and fat tails rather than just the mean and variance. A value above 1 means gains
outweigh losses at that threshold. A window with no downside (a zero denominator)
has an undefined ratio and returns `NaN`, as does a `NaN` return
(`nan_policy: ignore`).

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingOmega

    rng = np.random.default_rng(1)
    n = 500
    r = rng.standard_normal(n) * 0.01 + 0.0015          # a small positive drift
    omega = RollingOmega(60)(r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=omega, name='Omega', line=dict(color='steelblue')))
    fig.add_hline(y=1.0, line=dict(color='gray', dash='dot'),
                  annotation_text='gains = losses')
    fig.update_layout(title='RollingOmega: gains vs losses about the threshold (>1 favours gains)',
                      xaxis_title='bar', yaxis_title='Omega',
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
