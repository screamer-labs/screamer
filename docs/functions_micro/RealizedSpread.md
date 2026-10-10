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
- liquidity provision
- microstructure
short: 'Realized spread, 2*D*(trade price - mid lag steps later): the part of the effective
  spread the liquidity provider keeps.'
inputs: 2
outputs: 1
parameters:
- name: lag
  type: int
  default: 1
  min: 1
  description: Number of steps between a trade and the mid it is compared with (the price-impact
    horizon).
nan_policy: ignore
see_also:
- EffectiveSpread
- RollSpread
- MicroPrice
---

# `RealizedSpread`

## Description

`RealizedSpread` is the part of the effective spread that the liquidity provider
keeps once the price has moved. The inputs are the trade `price` and the `mid`
quote, in that order. With the quote-based trade direction
`D_t = sign(price_t - mid_t)` (a print above the mid is a buy, below a sell),

    realized_t = 2 * D_{t-lag} * (price_{t-lag} - mid_t).

It compares the trade from `lag` steps ago with the current mid, so it never uses
a future value. The difference `effective - realized` is the price-impact or
adverse-selection component: how far the mid moved in the trade's direction after
it printed. The first `lag` outputs are `NaN` (warmup).

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
