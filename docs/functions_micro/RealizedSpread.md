---
name: RealizedSpread
title: Realized Spread
implementation_family: micro
topics:
- price-impact
tags:
- realized spread
- price impact
- adverse selection
- transaction cost
- microstructure
short: "Realized spread, 2*D*(price - mid a few steps later): the liquidity part of the effective spread."
inputs: 2
outputs: 1
parameters:
- name: lag
  type: int
  default: 1
  min: 1
  description: How many steps ahead the mid is measured (the price-impact horizon).
nan_policy: ignore
see_also:
- EffectiveSpread
- RollSpread
---

# `RealizedSpread`

## Description

`RealizedSpread` is the part of the effective spread that the liquidity provider
keeps once the price has moved. With the quote-based trade direction (a print
above the mid is a buy, below is a sell), `D = sign(price - mid)`,

    realized = 2 * D * (price - mid measured `lag` steps later).

It compares a past trade's price to the mid `lag` steps afterward. What the
effective spread does not keep, `effective - realized`, is the price-impact or
adverse-selection component: how far the mid moved in the trade's direction after
it printed. The value at each step references a trade `lag` steps in the past, so
it is causal (never a future trade). The first `lag` samples are `NaN` (warmup).

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import EffectiveSpread, RealizedSpread

    rng = np.random.default_rng(2)
    n = 4000
    half_spread = 0.10
    mid = np.empty(n)
    mid[0] = 100.0
    price = np.empty(n)
    side = rng.choice([-1.0, 1.0], size=n)                 # trade direction
    for t in range(n):
        price[t] = mid[t] + side[t] * half_spread          # trade at bid/offer
        step = rng.standard_normal() * 0.01 + side[t] * 0.02   # adverse drift after the trade
        if t + 1 < n:
            mid[t + 1] = mid[t] + step

    lag = 20
    eff = EffectiveSpread()(price, mid)
    real = RealizedSpread(lag=lag)(price, mid)
    impact = eff - real                                    # price-impact / adverse selection

    fig = go.Figure()
    labels = ['effective spread', 'realized spread', 'price impact']
    series = [eff, real, impact]
    colors = ['crimson', 'seagreen', 'steelblue']
    for lab, s, c in zip(labels, series, colors):
        fig.add_trace(go.Bar(x=[lab], y=[np.nanmean(s)], marker_color=c, name=lab))
    fig.update_layout(title='Effective = realized + impact (means, lag=20)',
                      yaxis_title='spread', showlegend=False,
                      margin=dict(l=20, r=20, t=60, b=20))
    fig.show()
```
