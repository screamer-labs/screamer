---
name: RollingADF
title: Rolling augmented Dickey-Fuller statistic
implementation_family: fin
topics:
- regression
tags:
- adf
- dickey-fuller
- unit root
- cointegration
- stationarity
- pairs
- mean reversion
short: "Rolling augmented Dickey-Fuller t-statistic: how strongly a series, such as a pairs spread, reverts to a mean over the last window."
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 100
  min: 4
  description: Number of regression rows in the window. At least lags + 4.
- name: lags
  type: int
  default: 1
  min: 0
  description: Number of lagged differences in the regression (the augmentation). 0 is the plain Dickey-Fuller test.
nan_policy: ignore
see_also:
- RollingOU
- RollingHurst
- KalmanRegression
- RollingSpread
---

# `RollingADF`

## Description

`RollingADF(window_size, lags)(y)` fits, by ordinary least squares over the last
`window_size` rows,

    dy_t = c + gamma * y_{t-1} + phi_1 dy_{t-1} + ... + phi_p dy_{t-p} + e_t

with `p = lags`, and returns the t-statistic of `gamma`. A random walk has
`gamma = 0`; a series that reverts to a mean has `gamma < 0`, and the more
negative the statistic, the stronger the evidence against a unit root. The
lagged differences absorb short-term autocorrelation so it does not distort
the test.

Applied to the spread of a pair, it is the second step of the Engle-Granger
cointegration test: a pair is worth trading while the statistic stays below a
critical value. Under the null hypothesis the statistic does not follow the
normal distribution, so compare it with the Dickey-Fuller critical values
(MacKinnon 2010). For large windows they are about:

| | 1% | 5% | 10% |
|:--|--:|--:|--:|
| A series (constant, no trend) | -3.43 | -2.86 | -2.57 |
| The residual of a two-asset regression (Engle-Granger) | -3.90 | -3.34 | -3.04 |

Use the second row when the spread's hedge ratio was estimated on the same
data. Short windows have more negative critical values. The first row needs
`lags + 1` earlier values, so the output is `NaN` until step
`window_size + lags`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

A `NaN` input skips the step: the output is `NaN` and the window is unchanged.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingADF

    rng = np.random.default_rng(6)
    n = 1500
    phi = np.where(np.arange(n) < 750, 0.9, 1.0)       # mean-reverting, then a random walk
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi[t] * y[t - 1] + rng.standard_normal()

    adf = RollingADF(window_size=200, lags=1)(y)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('series', 'rolling ADF statistic'))
    fig.add_trace(go.Scatter(y=y, name='series'), row=1, col=1)
    fig.add_trace(go.Scatter(y=adf, name='ADF t-statistic'), row=2, col=1)
    fig.add_hline(y=-2.86, line_dash='dash', line_color='crimson', row=2, col=1,
                  annotation_text='5% critical value')
    fig.update_layout(title='The ADF statistic rises above the critical value when mean reversion stops',
                      margin=dict(l=20, r=20, t=80, b=20), showlegend=False)
    fig.show()
```

<!-- HELP_END -->
