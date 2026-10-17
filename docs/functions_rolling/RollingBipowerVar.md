---
name: RollingBipowerVar
title: Rolling Bipower Variation
implementation_family: rolling
topics:
- volatility
tags:
- bipower variation
- realized variance
- jumps
- jump-robust volatility
- barndorff-nielsen
- shephard
short: Jump-robust variance of a window of returns, (pi/2) times the sum of products of adjacent
  absolute returns, on the scale of the realized variance.
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
- RollingJumpShare
- RollingVar
- RollingRms
references:
- 'Barndorff-Nielsen, O. E. and Shephard, N. (2004). Power and bipower variation with stochastic
  volatility and jumps. *Journal of Financial Econometrics* 2(1), 1-37.'
---

# `RollingBipowerVar`

## Description

`RollingBipowerVar` is the bipower variation of a window of `n` returns
(Barndorff-Nielsen and Shephard 2004):

$$
BV = \frac{\pi}{2} \sum_{i=2}^{n} |r_i|\,|r_{i-1}| \cdot \frac{n}{n-1}
$$

The input is returns. The realized variance `RV = sum r_i^2` counts a jump at
its full size squared; in `BV` a jump is multiplied by its neighbours, which
are of normal size, so `BV` estimates the variance of the continuous part of
the returns. The factor `n / (n - 1)` scales the `n - 1` pairs to the `n` terms
of `RV`, so without jumps `BV` and `RV` estimate the same quantity.
[`RollingJumpShare`](RollingJumpShare.md) compares the two.

A `NaN` return is left out, and the returns on either side of it form a pair.

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
    from screamer import RollingBipowerVar, RollingSum

    rng = np.random.default_rng(2)
    r = rng.standard_normal(600) * 0.01
    r[[150, 300, 450]] = [0.08, -0.07, 0.09]            # three jumps
    bv = RollingBipowerVar(60)(r)
    rv = RollingSum(60)(r * r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=rv, name='realized variance', line=dict(color='gray')))
    fig.add_trace(go.Scatter(y=bv, name='bipower variation', line=dict(color='navy')))
    fig.update_layout(title='RollingBipowerVar: the jumps raise RV, not BV',
                      xaxis_title='step', yaxis_title='variance over 60 steps',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
