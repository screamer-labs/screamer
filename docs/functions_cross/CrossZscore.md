---
name: CrossZscore
title: Z-score of each asset within the row
implementation_family: cross
topics:
- cross-sectional
tags:
- z-score
- zscore
- standardize
- standard score
- cross-sectional z-score
- factor
- cross-sectional
short: 'Standardize each asset against the others at the same step: (x - mean) / std across
  the row.'
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- CrossRank
- CrossDemean
- CrossWinsorize
- RollingZscore
---

# `CrossZscore`

## Description

`CrossZscore` subtracts the row mean and divides by the row's sample standard
deviation (`ddof=1`), both taken over the assets that are not `NaN`. The result
says how many standard deviations each asset is above or below the others at
that step. A row whose valid values are all equal gives `0`, and a row with
one valid asset gives `NaN`.

Unlike ranks, z-scores keep the distances between assets, so one extreme value
can dominate. Apply [`CrossWinsorize`](CrossWinsorize.md) first when the
scores have outliers.

It is a cross-sectional operator: each event is one row of `K` values, one per
asset, and the output is a row of `K` values computed from that row alone. Call
it on a `(T, K)` array to process `T` rows, on one row of `K` values, or on an
iterator of rows to stream them. The number of assets is fixed by the first row
after `reset()`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` for an asset gives a `NaN` output for that asset at that step, and the statistic is computed across the remaining assets. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import CrossZscore

    rng = np.random.default_rng(3)
    names = ['A', 'B', 'C', 'D', 'E', 'F']
    returns = rng.standard_normal((250, 6)) * 0.01 + np.linspace(-4e-4, 4e-4, 6)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    momentum = np.log(prices[60:] / prices[:-60])         # 60-day return per asset
    z = CrossZscore()(momentum)

    fig = go.Figure()
    for k, name in enumerate(names):
        fig.add_trace(go.Scatter(y=z[:, k], mode='lines', name=name))
    fig.update_layout(title='CrossZscore of 60-day momentum across six assets',
                      xaxis_title='Step', yaxis_title='Z-score within the row',
                      margin=dict(l=20, r=20, t=80, b=20))
    fig.show()
```

<!-- HELP_END -->
