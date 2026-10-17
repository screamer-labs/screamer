---
name: BayesianChangepoint
title: Bayesian online changepoint detection
implementation_family: fin
topics:
- regime
tags:
- changepoint
- change point
- structural break
- bayesian
- adams-mackay
- run length
- regime change
short: Online Bayesian changepoint detection (Adams-MacKay) on a series with Gaussian segments
  of unknown mean and variance; the expected run length since the last change and a predictive
  mean and standard deviation.
inputs: 1
outputs: 3
parameters:
- name: expected_run
  type: float
  default: 250.0
  min: 1.0
  description: Expected number of steps between changes; the hazard of a change at each step
    is 1 / expected_run.
- name: max_run
  type: int
  default: 1000
  min: 1
  description: Longest run length tracked. Longer runs are dropped, which bounds the cost per
    step.
- name: prior_var
  type: float
  default: 1.0
  description: Scale of the prior on a segment's variance (the normal-gamma beta). Must be
    positive. 1 suits a standardized input.
nan_policy: ignore
see_also:
- CusumFilter
- RollingHMM
- RollingZscore
- EwStd
output_names:
- run_length
- pred_mean
- pred_std
references:
- 'Adams, R. P. and MacKay, D. J. C. (2007). Bayesian online changepoint detection. arXiv:0710.3742.'
---

# `BayesianChangepoint`

## Description

`BayesianChangepoint` is the online Bayesian changepoint detector of Adams and
MacKay (2007). It models the input as segments; within a segment the values
are Gaussian with a mean and variance it does not know, and at each step a new
segment starts with probability `1 / expected_run`. After each value it updates
the probability of every run length `r`, the number of steps since the last
change (0 on the step of a change), and returns three columns:

0. `run_length`: the posterior expected run length.
1. `pred_mean`: the mean of the predictive distribution of the next value,
   mixed over the current run lengths, given no change at the next step.
2. `pred_std`: the standard deviation of that mixture.

Each segment's mean and variance have a normal-gamma prior with mean 0,
`kappa=1`, `alpha=1`, and `beta=prior_var`, so the predictive of a run is a
Student-t. Feed a standardized series, such as a `RollingZscore` of returns or
returns divided by an `EwStd` of them: the prior assumes values of order 1. A
drop of `run_length` towards 0 marks a change; after a shift of the mean by 3
standard deviations it falls within a few steps.

A run longer than `max_run` is dropped and the rest renormalized, so
`run_length` never exceeds `max_run` and each step costs at most
`max_run + 1` updates.

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
    from screamer import BayesianChangepoint

    rng = np.random.default_rng(7)
    x = np.r_[rng.standard_normal(300), 2.5 + rng.standard_normal(300),
              0.5 * rng.standard_normal(300)]
    out = BayesianChangepoint(expected_run=200.0)(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=x, name='input', line=dict(color='lightgray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=out[:, 1], name='pred_mean', line=dict(color='navy')), row=1, col=1)
    fig.add_trace(go.Scatter(y=out[:, 0], name='run_length', line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(title='BayesianChangepoint: the run length restarts at each change',
                      xaxis2_title='step', margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
