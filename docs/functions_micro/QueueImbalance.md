---
name: QueueImbalance
title: Queue (Book) Imbalance
implementation_family: micro
topics:
- order-flow-imbalance
tags:
- queue imbalance
- book imbalance
- order book imbalance
- order flow
- microstructure
short: "Normalized L1 book imbalance, (bid_size - ask_size) / (bid_size + ask_size)."
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- OFI
- MicroPrice
---

# `QueueImbalance`

## Description

`QueueImbalance` is the L1 order-book (queue) imbalance,

    (bid_size - ask_size) / (bid_size + ask_size),

in `[-1, 1]`. It is positive when more size rests on the bid than the ask
(buy pressure) and negative when the ask is heavier. It is one of the most
widely used short-horizon predictors: a heavy bid queue tends to precede an
up-move. Feed the resting sizes at the top of book: `QueueImbalance(bid_size,
ask_size)`. An empty book (both sizes 0) has no imbalance and returns 0.

`QueueImbalance` is a documented synonym of `OFI`: the same normalized-imbalance
operator, applied to resting queue sizes rather than to trade flow.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer.microstructure import QueueImbalance

    rng = np.random.default_rng(2)
    n = 200
    wave = np.sin(np.linspace(0, 4 * np.pi, n))
    bid_size = np.abs(rng.standard_normal(n)) + 1 + 2 * wave.clip(0)
    ask_size = np.abs(rng.standard_normal(n)) + 1 + 2 * (-wave).clip(0)
    imbalance = QueueImbalance()(bid_size, ask_size)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                        vertical_spacing=0.08)
    fig.add_trace(go.Bar(y=bid_size, name='bid size', marker_color='seagreen'), row=1, col=1)
    fig.add_trace(go.Bar(y=-ask_size, name='ask size', marker_color='crimson'), row=1, col=1)
    fig.add_trace(go.Scatter(y=imbalance, name='queue imbalance', line=dict(color='navy')),
                  row=2, col=1)
    fig.update_layout(barmode='relative',
                      title='QueueImbalance: L1 book pressure in [-1, 1]',
                      yaxis=dict(title='resting size'), yaxis2=dict(title='imbalance', range=[-1, 1]),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
