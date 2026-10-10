---
name: OFI
title: Order-Flow Imbalance
implementation_family: micro
topics:
- order-flow-imbalance
tags:
- order flow imbalance
- order imbalance
- trade imbalance
- ofi
- buy sell imbalance
- order flow
- microstructure
short: Normalized signed order flow, (buy - sell) / (buy + sell).
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- ContOFI
- QueueImbalance
- RollingOrderImbalance
- VPIN
- SignedVolume
---

# `OFI`

## Description

`OFI` is the trade-flow order imbalance: the net of buy-aggressor and
sell-aggressor volume, normalized by their total. The inputs are `buy` and
`sell` volume, in that order:

    OFI = (buy - sell) / (buy + sell).

It lies in `[-1, 1]`, is positive when buyers lift the offer more than sellers
hit the bid, and is `0` on an empty bucket (both volumes zero). For the
order-book version, computed from changes in resting depth, use
[`ContOFI`](ContOFI.md).

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
