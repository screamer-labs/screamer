---
name: CrossNormalize
title: Scale a row of scores into portfolio weights
implementation_family: cross
topics:
- cross-sectional
tags:
- weights
- normalize
- portfolio weights
- gross exposure
- net exposure
- leverage
- dollar neutral
- portfolio
- cross-sectional
short: Scale each row of scores to portfolio weights with a chosen gross exposure, and optionally
  a net exposure.
inputs: 1
outputs: 1
parameters:
- name: gross
  type: float
  default: 1.0
  description: Sum of the absolute weights of each row.
- name: net
  type: float|null
  default: null
  description: Sum of the weights of each row, in [-gross, gross]. None (the default) keeps
    the signs of the scores and sets no net target.
nan_policy: ignore
see_also:
- CrossRank
- CrossDemean
- CrossTopBottom
---

# `CrossNormalize`

## Description

`CrossNormalize` turns a row of scores into weights.

- With the default `net=None`, it scales the row so the absolute weights sum to
  `gross`, keeping each score's sign: `w = x * gross / sum(|x|)`.
- With `net` set, it first subtracts the row mean, then scales the positive
  side to sum to `(gross + net) / 2` and the negative side to
  `(gross - net) / 2`. The weights then have gross exposure `gross` and net
  exposure `net`; `net=0` is dollar-neutral.

A row with nothing to scale (all zeros, or all equal with a net target) gives
zero weights. Combine it with [`CrossRank`](CrossRank.md) for a rank-weighted
long/short portfolio.

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
    from screamer import CrossNormalize, CrossRank

    rng = np.random.default_rng(3)
    names = ['A', 'B', 'C', 'D', 'E', 'F']
    returns = rng.standard_normal((250, 6)) * 0.01 + np.linspace(-4e-4, 4e-4, 6)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    momentum = np.log(prices[60:] / prices[:-60])         # 60-day return per asset
    weights = CrossNormalize(gross=1.0, net=0.0)(CrossRank()(momentum))

    fig = go.Figure()
    for k, name in enumerate(names):
        fig.add_trace(go.Scatter(y=weights[:, k], mode='lines', name=name))
    fig.add_trace(go.Scatter(y=np.nansum(np.abs(weights), axis=1), mode='lines', name='gross',
                             line=dict(color='black', dash='dash')))
    fig.update_layout(title='Dollar-neutral rank weights with gross exposure 1',
                      xaxis_title='Step', yaxis_title='Weight',
                      margin=dict(l=20, r=20, t=80, b=20))
    fig.show()
```

<!-- HELP_END -->
