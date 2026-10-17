---
name: RollingJumpShare
title: Rolling Jump Share
implementation_family: rolling
topics:
- volatility
- regime
tags:
- jumps
- jump share
- jump test
- bipower variation
- realized variance
short: 'Share of the realized variance of a window of returns that comes from jumps: max(0, 1 - BV / RV).'
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent returns in the window.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" uses the
    returns seen so far (at least two), and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- RollingBipowerVar
- RollingVar
- RollingVarianceRatio
references:
- 'Barndorff-Nielsen, O. E. and Shephard, N. (2004). Power and bipower variation with stochastic
  volatility and jumps. *Journal of Financial Econometrics* 2(1), 1-37.'
---

# `RollingJumpShare`

## Description

`RollingJumpShare` is `max(0, 1 - BV / RV)`, with `BV` the bipower variation of [`RollingBipowerVar`](RollingBipowerVar.md) and `RV = sum r_i^2` the realized variance of the same window. It is the share of the window's variance that comes from jumps (Barndorff-Nielsen and Shephard 2004): near 0 for diffusive returns and near 1 when a few returns carry most of the variance. On a window of returns `[1, 1, 1, 10]`, `BV = 8 pi`, `RV = 103`, and the share is `0.756`. It is 0 when `RV` is 0, and 0 when `BV` exceeds `RV`, which sampling noise causes on windows without jumps.

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
    from screamer import RollingJumpShare

    rng = np.random.default_rng(2)
    r = rng.standard_normal(600) * 0.01
    r[[150, 300, 450]] = [0.08, -0.07, 0.09]            # three jumps
    share = RollingJumpShare(60)(r)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=r, name='returns', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=share, name='jump share', line=dict(color='navy')), row=2, col=1)
    fig.update_layout(title='RollingJumpShare: the share of variance from jumps',
                      xaxis2_title='step',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
