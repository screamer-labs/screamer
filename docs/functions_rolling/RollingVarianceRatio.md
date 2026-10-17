---
name: RollingVarianceRatio
title: Rolling Variance Ratio
implementation_family: rolling
topics:
- statistics
- regime
tags:
- variance ratio
- lo-mackinlay
- random walk test
- autocorrelation
- trend
- mean reversion
- efficiency
short: Lo-MacKinlay variance ratio of a window of returns, near 1 for a random walk, above 1
  when returns trend and below 1 when they revert, or its robust z-statistic.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 256
  min: 5
  description: Number of most recent returns in the window. Must be more than 2 lag.
- name: lag
  type: int
  default: 2
  min: 2
  description: Number of returns summed into each long-horizon return (q).
- name: output
  type: str
  default: ratio
  enum:
  - ratio
  - zscore
  description: '"ratio" for the variance ratio VR(q), "zscore" for the heteroskedasticity-robust
    statistic z*(q).'
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" uses the
    returns seen so far once there are more than 2 lag, and "zero" treats the window as pre-filled
    with zeros.'
nan_policy: ignore
see_also:
- RollingHurst
- RollingOU
references:
- 'Lo, A. W. and MacKinlay, A. C. (1988). Stock market prices do not follow random walks: evidence
  from a simple specification test. *Review of Financial Studies* 1(1), 41-66.'
---

# `RollingVarianceRatio`

## Description

`RollingVarianceRatio` is the variance ratio of Lo and MacKinlay (1988) on a
trailing window of `n` returns: the variance of the overlapping `q`-step sums
of returns, divided by `q` times the variance of one-step returns, with
`q = lag`. Returns that are uncorrelated give a ratio near 1; returns that
continue in the same direction (a trend) give more than 1; returns that
reverse (mean reversion) give less than 1.

With `mu` the window mean, the one-step variance is
`sum (r_i - mu)^2 / (n - 1)` and the `q`-step variance is the sum of the
squared deviations of the `n - q + 1` overlapping `q`-sums from `q mu`,
divided by `m = q (n - q + 1)(1 - q / n)`; both are the authors'
bias-corrected estimators. On the returns `[1, -1, 1, -1, 1]` with `lag=2`,
every 2-sum is 0 and the ratio is `1/9`.

`output="zscore"` gives the heteroskedasticity-robust statistic `z*(q)`, which
is approximately standard normal when the returns are uncorrelated, even with
changing volatility; `|z*| > 1.96` rejects a random walk at the 5% level. The
output is `NaN` on a window whose returns are all equal.

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
    from screamer import RollingVarianceRatio

    rng = np.random.default_rng(4)
    e = rng.standard_normal(3000)
    r = np.empty(3000)
    r[0] = e[0]
    for t in range(1, 3000):
        phi = 0.3 if t < 1500 else -0.3         # trending, then reverting
        r[t] = phi * r[t - 1] + e[t]
    vr = RollingVarianceRatio(250, lag=5)(r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=vr, name='VR(5)', line=dict(color='navy')))
    fig.add_hline(y=1.0, line=dict(color='gray', dash='dot'))
    fig.update_layout(title='RollingVarianceRatio: above 1 while returns trend, below 1 while they revert',
                      xaxis_title='step', yaxis_title='variance ratio',
                      margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
    fig.show()
```

<!-- HELP_END -->
