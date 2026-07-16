---
name: MicroPrice
title: Micro-Price (imbalance-weighted mid)
implementation_family: micro
topics:
- price-impact
tags:
- micro-price
- microprice
- fair value
- weighted mid
- stoikov
- order book
- microstructure
short: "Imbalance-weighted mid (Stoikov 2018, first-order): fair value that leans toward the thinner side of the book."
inputs: 4
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- OFI
- QueueImbalance
---

# `MicroPrice`

## Description

`MicroPrice` is the imbalance-weighted mid, a fair value that leans toward the
thinner side of the order book (Stoikov 2018, first-order form). With
`I = bid_size / (bid_size + ask_size)` the fraction of resting size on the bid,

    micro = I * ask + (1 - I) * bid.

When the bid queue is heavier (`I` near 1) the price is pulled toward the ask
(upward pressure); when the ask queue is heavier it is pulled toward the bid. A
balanced or empty book gives the plain mid. `MicroPrice(bid, ask, bid_size,
ask_size)` predicts the next mid better than the plain mid because it uses the
queue imbalance. This is the widely-used weighted-mid form; the full Stoikov
micro-price with a calibrated adjustment function is not modelled here.
References: Stoikov (2018), "The micro-price: a high-frequency estimator of future
prices".

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import MicroPrice

    rng = np.random.default_rng(1)
    n = 200
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.02)
    bid, ask = mid - 0.25, mid + 0.25          # a 0.5-wide spread
    bid_size = np.abs(rng.standard_normal(n)) + 0.5
    ask_size = np.abs(rng.standard_normal(n)) + 0.5
    micro = MicroPrice()(bid, ask, bid_size, ask_size)
    imbalance = (bid_size - ask_size) / (bid_size + ask_size)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=micro - mid, mode='lines', name='micro-price - mid',
                             line=dict(color='teal')), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='gray', dash='dot'), row=1, col=1)
    fig.add_trace(go.Scatter(y=imbalance, mode='lines', name='queue imbalance',
                             line=dict(color='steelblue')), row=2, col=1)
    fig.update_layout(title='MicroPrice: fair value leans toward the heavier queue',
                      yaxis=dict(title='micro - mid'), yaxis2=dict(title='imbalance', range=[-1, 1]),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
