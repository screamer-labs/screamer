---
name: RollingUpsideDeviation
title: Rolling Upside Deviation
implementation_family: fin
topics:
- risk
- volatility
tags:
- upside deviation
- upside risk
- semideviation
- upside volatility
- gain-loss ratio
short: 'Trailing-window upside semideviation: the RMS of the returns above a minimum acceptable
  return.'
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent returns in the window.
- name: mar
  type: float
  default: 0.0
  description: Minimum acceptable return. Only returns above it contribute to the deviation.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- RollingDownsideDeviation
- RollingStd
- RollingSortino
- RollingOmega
---

# `RollingUpsideDeviation`

## Description

`RollingUpsideDeviation` is the upside semideviation over a trailing window:
the root-mean-square of the excesses above a minimum acceptable return `mar`,

    sqrt( mean( max(return - mar, 0)^2 ) ).

Only returns above `mar` contribute; anything at or below it counts as zero.
It is the mirror of [`RollingDownsideDeviation`](RollingDownsideDeviation.md):
`RollingUpsideDeviation(w, mar)(r)` equals `RollingDownsideDeviation(w, -mar)(-r)`.
The ratio of the two measures the asymmetry of a return distribution, for
example a strategy whose gains are larger than its losses.

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
    from screamer import RollingDownsideDeviation, RollingUpsideDeviation

    rng = np.random.default_rng(0)
    n = 500
    # returns with occasional sharp gains (right skew)
    r = rng.standard_normal(n) * 0.01 + (rng.random(n) < 0.05) * 0.03
    up = RollingUpsideDeviation(60)(r)
    down = RollingDownsideDeviation(60)(r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=up, name='upside deviation', line=dict(color='seagreen')))
    fig.add_trace(go.Scatter(y=down, name='downside deviation', line=dict(color='crimson')))
    fig.update_layout(title='RollingUpsideDeviation: the gains above mar, against the losses below it',
                      xaxis_title='step', yaxis_title='per-step deviation',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
