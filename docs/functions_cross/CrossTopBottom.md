---
name: CrossTopBottom
title: Long the top, short the bottom of each row
implementation_family: cross
topics:
- cross-sectional
tags:
- long short
- top bottom
- quantile portfolio
- decile portfolio
- selection
- cross-sectional
short: +1 for the highest values of each row, -1 for the lowest, 0 for the rest.
inputs: 1
outputs: 1
parameters:
- name: count
  type: int
  default: 1
  min: 1
  description: Number of assets on each side.
nan_policy: ignore
see_also:
- CrossRank
- CrossNormalize
---

# `CrossTopBottom`

## Description

`CrossTopBottom` selects the `count` assets with the highest values of each row
(`+1`) and the `count` with the lowest (`-1`), and gives `0` to the rest. Ties
are broken by column order: of two equal values, the earlier column ranks
lower. With fewer than `2 * count` valid assets, `count` is reduced to half of
them so no asset is on both sides, and a row whose valid values are all equal
gives `0` for every asset.

Pass the result to [`CrossNormalize`](CrossNormalize.md) for equal weights per
side.

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
    from screamer import CrossTopBottom

    rng = np.random.default_rng(3)
    names = ['A', 'B', 'C', 'D', 'E', 'F']
    returns = rng.standard_normal((250, 6)) * 0.01 + np.linspace(-4e-4, 4e-4, 6)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    momentum = np.log(prices[60:] / prices[:-60])         # 60-day return per asset
    sides = CrossTopBottom(count=2)(momentum)

    fig = go.Figure(go.Heatmap(z=sides.T, y=names, colorscale=[[0, 'crimson'], [0.5, 'white'], [1, 'seagreen']],
                               zmin=-1, zmax=1, showscale=False))
    fig.update_layout(title='CrossTopBottom(count=2): long the two strongest, short the two weakest',
                      xaxis_title='Step', margin=dict(l=20, r=20, t=80, b=20))
    fig.show()
```

<!-- HELP_END -->
