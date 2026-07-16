---
name: EffectiveSpread
title: Effective Spread
implementation_family: micro
topics:
- price-impact
tags:
- effective spread
- transaction cost
- spread
- liquidity
- microstructure
short: "Effective spread, 2*|price - mid|: the round-trip cost paid relative to the mid."
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- RealizedSpread
- RollSpread
---

# `EffectiveSpread`

## Description

`EffectiveSpread` is twice the absolute distance of the trade price from the
mid-quote at trade time,

    effective = 2 * |price - mid|.

It is the round-trip cost actually paid relative to the mid: a buy lifts the
offer and a sell hits the bid, so the print sits away from the mid by about half
the spread. Unlike the quoted spread it reflects where trades really occur,
including price improvement and trading through the quote. Pair it with
[`RealizedSpread`](RealizedSpread.md) to split that cost into a realized
(liquidity) part and a price-impact (adverse-selection) part.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import EffectiveSpread

    rng = np.random.default_rng(1)
    n = 250
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.02)
    half_spread = 0.10
    price = mid + rng.choice([-half_spread, half_spread], size=n)   # trades at bid or offer
    eff = EffectiveSpread()(price, mid)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=mid, mode='lines', name='mid', line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=price, mode='markers', name='trade price',
                             marker=dict(color='lightslategray', size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(y=eff, mode='lines', name='effective spread',
                             line=dict(color='crimson')), row=2, col=1)
    fig.add_hline(y=2 * half_spread, line=dict(color='gray', dash='dot'), row=2, col=1)
    fig.update_layout(title='EffectiveSpread: 2*|price - mid|, the cost paid per trade',
                      yaxis=dict(title='price'), yaxis2=dict(title='eff. spread'),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
