---
name: CrossDemean
title: Subtract the row mean
implementation_family: cross
topics:
- cross-sectional
tags:
- demean
- market neutral
- cross-sectional
short: "Subtract the mean across assets at each step, removing the common component."
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- CrossZscore
- CrossNormalize
---

# `CrossDemean`

## Description

`CrossDemean` subtracts the mean of the row, taken over the assets that are not
`NaN`, from every asset. What is left is each asset's value relative to the
others. Applied to returns it removes the move the assets share, and applied
to scores it gives weights that sum to zero, a starting point for a
market-neutral book.

It is a cross-sectional operator: each event is one row of `K` values, one per
asset, and the output is a row of `K` values computed from that row alone. Call
it on a `(T, K)` array to process `T` rows, on one row of `K` values, or on an
iterator of rows to stream them. The number of assets is fixed by the first row
after `reset()`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

A `NaN` entry is an asset that is not available on that row, for example one
that is not listed yet or is halted. It is left out of the cross-section, and
its output is `NaN`. A row that is all `NaN` returns all `NaN`.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import CrossDemean

    rng = np.random.default_rng(3)
    names = ['A', 'B', 'C', 'D', 'E', 'F']
    returns = rng.standard_normal((250, 6)) * 0.01 + np.linspace(-4e-4, 4e-4, 6)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    momentum = np.log(prices[60:] / prices[:-60])         # 60-day return per asset
    relative = CrossDemean()(returns)                   # each asset's return minus the average
    relative_prices = 100 * np.exp(np.cumsum(relative, axis=0))

    fig = go.Figure()
    for k, name in enumerate(names):
        fig.add_trace(go.Scatter(y=relative_prices[:, k], mode='lines', name=name))
    fig.update_layout(title='Performance relative to the average of the six assets',
                      xaxis_title='Step', yaxis_title='Relative price',
                      margin=dict(l=20, r=20, t=80, b=20))
    fig.show()
```

<!-- HELP_END -->
