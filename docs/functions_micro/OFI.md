---
name: OFI
title: Order-Flow Imbalance
implementation_family: micro
topics:
- microstructure
tags:
- order flow
- imbalance
- ofi
- flow
- microstructure
short: Normalized signed order flow, (buy - sell) / (buy + sell).
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `OFI`

## Description

`OFI` is the order-flow imbalance: the net of buy-aggressor and sell-aggressor
volume, normalized by their total. It lives in `[-1, 1]`, is positive when buyers
lift the offer more than sellers hit the bid, and is `0` on an empty bucket.

Order-flow imbalance is the standard short-horizon driver of price: over a bar it
explains a large share of the price change (see `RollingKyleLambda`). Feed
`OFI(buy_volume, sell_volume)` as a signal, or pair it with returns to estimate
price impact. References: Cont, Kukanov, Stoikov (2014), "The price impact of
order book events".

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import OFI

    rng = np.random.default_rng(4)
    n = 200
    wave = np.sin(np.linspace(0, 4 * np.pi, n))          # slow shift in pressure
    buy = np.abs(rng.standard_normal(n)) + 1 + 2 * wave.clip(0)
    sell = np.abs(rng.standard_normal(n)) + 1 + 2 * (-wave).clip(0)
    ofi = OFI()(buy, sell)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                        vertical_spacing=0.08)
    fig.add_trace(go.Bar(y=buy, name='buy volume', marker_color='seagreen'), row=1, col=1)
    fig.add_trace(go.Bar(y=-sell, name='sell volume', marker_color='crimson'), row=1, col=1)
    fig.add_trace(go.Scatter(y=ofi, name='OFI', line=dict(color='navy')), row=2, col=1)
    fig.update_layout(barmode='relative',
                      title='OFI: order-flow imbalance in [-1, 1]',
                      yaxis=dict(title='volume'), yaxis2=dict(title='OFI', range=[-1, 1]),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
