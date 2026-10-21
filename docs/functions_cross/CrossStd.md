---
name: CrossStd
title: Standard deviation across the row
implementation_family: cross
topics:
- cross-sectional
tags:
- dispersion
- cross-sectional dispersion
- cross-sectional standard deviation
- std
- standard deviation
- return dispersion
- cross-sectional
short: 'Standard deviation across the assets of each step: one dispersion value per row.'
inputs: 1
outputs: 1
parameters:
- name: ddof
  type: int
  default: 1
  min: 0
  description: Delta degrees of freedom. The divisor is n - ddof, with n the number of non-NaN assets in the row. 1 gives the sample standard deviation (pandas), 0 the population one (numpy).
nan_policy: ignore
see_also:
- CrossZscore
- CrossDemean
- RollingStd
---

# `CrossStd`

## Description

`CrossStd` returns the standard deviation of each row's values, taken over the
assets that are not `NaN`. With `n` valid assets and row mean $\bar{x}_t$,

$$
y[t] = \sqrt{\frac{1}{n - \text{ddof}} \sum_{i \text{ valid}} \left(x_{t,i} - \bar{x}_t\right)^2}
$$

It matches `pandas.DataFrame.std(axis=1, ddof=ddof)` and
`numpy.nanstd(x, axis=1, ddof=ddof)`. The default `ddof=1` is the sample
standard deviation, the scale [`CrossZscore`](CrossZscore.md) divides by. A
row with `n <= ddof` valid assets gives `NaN`.

Applied to returns, `y[t]` is the cross-sectional return dispersion: how far
apart the assets moved at step `t`. Dispersion is used to condition momentum
and other cross-sectional strategies, and to scale cross-sectional scores.

`CrossStd` is a cross-sectional reduction: each event is one row of `K`
values, one per asset, and the output is one value computed from that row
alone. A `(T, K)` array gives a `(T,)` series, one row of `K` values gives a
float, and an iterator of rows gives an iterator of floats. The number of
assets is fixed by the first row after `reset()`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` for an asset leaves that asset out of the row, and the statistic is computed across the remaining assets. A row with too few valid assets gives `NaN`. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import CrossStd, LogReturn

    rng = np.random.default_rng(5)
    vol = np.where(np.arange(500) < 250, 0.01, 0.025)[:, None]   # calm, then turbulent
    returns = rng.standard_normal((500, 8)) * vol
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    dispersion = CrossStd()(LogReturn()(prices))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for k in range(prices.shape[1]):
        fig.add_trace(go.Scatter(y=prices[:, k], mode='lines', line=dict(width=1),
                                 showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(y=dispersion, mode='lines', name='CrossStd of returns'),
                  row=2, col=1)
    fig.update_layout(title='Return dispersion across eight assets',
                      xaxis2_title='Step', yaxis_title='Price', yaxis2_title='Dispersion',
                      margin=dict(l=20, r=20, t=80, b=20))
    fig.show()
```

<!-- HELP_END -->
