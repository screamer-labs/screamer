---
name: CrossRank
title: Rank of each asset within the row
implementation_family: cross
topics:
- cross-sectional
tags:
- rank
- cross-sectional rank
- percentile rank
- rank weights
- portfolio
- factor
- cross-sectional
short: Rank each asset against the others at the same step, scaled to [-1, 1] or [0, 1].
inputs: 1
outputs: 1
parameters:
- name: scale
  type: str
  default: signed
  description: 'Output range: "signed" maps the lowest value to -1 and the highest to 1; "unit"
    maps them to 0 and 1.'
nan_policy: ignore
see_also:
- CrossZscore
- CrossTopBottom
- CrossNormalize
- RollingRank
---

# `CrossRank`

## Description

`CrossRank` ranks the assets of each row by value. Equal values share their
average rank. The ranks are scaled so the lowest value maps to `-1` and the
highest to `1` (`scale="signed"`, the default), or to `0` and `1`
(`scale="unit"`). A row with one valid asset gives the midpoint, `0` or `0.5`.

Ranks are robust to outliers and to the scale of the scores, so they are a
common first step in turning a factor score into portfolio weights:
`CrossNormalize` then scales the ranks to a gross exposure.

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
    from screamer import CrossRank

    rng = np.random.default_rng(3)
    names = ['A', 'B', 'C', 'D', 'E', 'F']
    returns = rng.standard_normal((250, 6)) * 0.01 + np.linspace(-4e-4, 4e-4, 6)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    momentum = np.log(prices[60:] / prices[:-60])         # 60-day return per asset
    ranks = CrossRank()(momentum)                       # (T, 6), in [-1, 1]

    fig = go.Figure()
    for k, name in enumerate(names):
        fig.add_trace(go.Scatter(y=ranks[:, k], mode='lines', name=name))
    fig.update_layout(title='CrossRank of 60-day momentum across six assets',
                      xaxis_title='Step', yaxis_title='Signed rank',
                      margin=dict(l=20, r=20, t=80, b=20))
    fig.show()
```

<!-- HELP_END -->
