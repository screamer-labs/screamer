---
name: CrossWinsorize
title: Clip each row to its quantiles
implementation_family: cross
topics:
- cross-sectional
tags:
- winsorize
- winsorise
- outliers
- clip
- clamp
- cross-sectional
short: Clip each asset's value to the lower and upper quantiles of its row.
inputs: 1
outputs: 1
parameters:
- name: lower
  type: float
  default: 0.05
  description: Lower quantile in [0, 1].
- name: upper
  type: float
  default: 0.95
  description: Upper quantile in [lower, 1].
nan_policy: ignore
see_also:
- CrossZscore
- Clip
- Hampel
---

# `CrossWinsorize`

## Description

`CrossWinsorize` computes the `lower` and `upper` quantiles of each row, over the
assets that are not `NaN`, and clips every value to that range. The quantiles
use linear interpolation between order statistics, the same as numpy's default
`quantile`. Use it before [`CrossZscore`](CrossZscore.md) or
[`CrossNormalize`](CrossNormalize.md) so one extreme score does not dominate
the weights.

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
    from screamer import CrossWinsorize

    rng = np.random.default_rng(3)
    names = ['A', 'B', 'C', 'D', 'E', 'F']
    returns = rng.standard_normal((250, 6)) * 0.01 + np.linspace(-4e-4, 4e-4, 6)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    momentum = np.log(prices[60:] / prices[:-60])         # 60-day return per asset
    scores = momentum.copy()
    scores[100, 2] = 1.5                                # one bad data point
    clipped = CrossWinsorize(0.1, 0.9)(scores)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scores[:, 2], mode='lines', name='C, raw'))
    fig.add_trace(go.Scatter(y=clipped[:, 2], mode='lines', name='C, winsorized'))
    fig.update_layout(title='CrossWinsorize limits a bad value to the range of the other assets',
                      xaxis_title='Step', yaxis_title='Score',
                      margin=dict(l=20, r=20, t=80, b=20))
    fig.show()
```

<!-- HELP_END -->
