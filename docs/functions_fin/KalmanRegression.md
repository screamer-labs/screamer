---
name: KalmanRegression
title: Kalman filter regression (dynamic hedge ratio)
implementation_family: fin
topics:
- regression
tags:
- kalman
- hedge ratio
- pairs
- regression
- time-varying beta
short: "Regression with a random-walk intercept and slope, estimated by a Kalman filter: the dynamic hedge ratio of a pairs trade, its spread, and the spread's predicted deviation."
inputs: 2
outputs: 4
parameters:
- name: delta
  type: float
  default: 1.0e-06
  description: How fast the intercept and slope may move. The state noise variance is delta / (1 - delta) per step; smaller values give steadier estimates and a spread that keeps its mean reversion.
- name: obs_var
  type: float
  default: 0.001
  description: Variance of the spread (the regression error), in squared units of the first input.
- name: prior_var
  type: float
  default: 1.0
  min: 0.0
  description: Starting variance of the intercept and slope, which start at zero.
nan_policy: ignore
see_also:
- RollingBeta
- RollingTLSBeta
- BayesianRegression
- KalmanFilter
- BandSignal
---

# `KalmanRegression`

## Description

`KalmanRegression(delta, obs_var)(y, x)` fits `y_t = alpha_t + beta_t * x_t + e_t`
where `alpha` and `beta` follow random walks, and updates the estimate at every
step with a Kalman filter. In a pairs trade `y` and `x` are the log prices of
the two legs, `beta` is the hedge ratio, and `e` is the spread (Chan 2013,
chapter 3). Unlike a rolling regression, the estimate has no window: it weighs
the past with a decay that `delta` sets, and adapts at once when the
relationship moves.

It returns four columns:

0. `beta`: the slope after the step's update.
1. `alpha`: the intercept after the update.
2. `spread`: the prediction error `y - (alpha + beta * x)` made with the
   previous step's estimate, so the hedge ratio it uses does not depend on
   the current prices.
3. `spread_std`: the predicted standard deviation of that error. With
   `obs_var` near the variance of the spread, `spread / spread_std` is a
   standardized spread for a band rule such as
   [`BandSignal`](../functions_signal/BandSignal.md).

`delta` is the main choice. A large value lets the intercept follow the spread
itself, so the spread loses the mean reversion the trade relies on; on log
prices values from `1e-8` to `1e-5` keep the hedge ratio steady while it still
adapts. Set `obs_var` near the variance of the spread.

The intercept and slope start at zero with variance `prior_var`. The default
of 1 is weak enough that the data sets the line within a few steps, and keeps
the first updates numerically well conditioned. The spread and its deviation
are `NaN` on the first step, which has no prediction. The first input is
the dependent series, as in [`RollingBeta`](RollingBeta.md).

<!-- NAN_FOOTNOTE_START -->
## NaN handling

A `NaN` in either input skips the step: the four outputs are `NaN` and the estimate is unchanged.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import KalmanRegression

    rng = np.random.default_rng(4)
    n = 1500
    common = np.cumsum(rng.standard_normal(n)) * 0.02            # shared random walk
    hedge = np.linspace(0.8, 1.2, n)                              # slowly changing hedge ratio
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.97 * spread[t - 1] + rng.standard_normal() * 0.004   # mean-reverting
    log_b = 4.0 + common
    log_a = 0.5 + hedge * log_b + spread

    out = KalmanRegression(delta=1e-6, obs_var=3e-4)(log_a, log_b)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('hedge ratio', 'spread z-score'))
    fig.add_trace(go.Scatter(y=hedge, name='true hedge ratio', line=dict(dash='dash', color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=out[:, 0], name='KalmanRegression'), row=1, col=1)
    fig.add_trace(go.Scatter(y=out[:, 2] / out[:, 3], name='spread / spread_std'), row=2, col=1)
    fig.update_yaxes(range=[0.6, 1.4], row=1, col=1)
    fig.add_hline(y=0, line_color='gray', row=2, col=1)
    fig.update_layout(title='KalmanRegression tracks a drifting hedge ratio', margin=dict(l=20, r=20, t=80, b=20))
    fig.show()
```

<!-- HELP_END -->
