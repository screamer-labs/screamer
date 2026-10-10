---
name: ContOFI
title: Cont Order-Flow Imbalance (order book)
implementation_family: micro
topics:
- order-flow-imbalance
tags:
- order flow imbalance
- ofi
- order book
- book events
- l1
- cont
- cont-kukanov-stoikov
- microstructure
short: Cont-Kukanov-Stoikov (2014) order-flow imbalance from L1 book events.
inputs: 4
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- OFI
- QueueImbalance
- MicroPrice
- RollingOrderImbalance
references:
- Cont, R., Kukanov, A. and Stoikov, S. (2014). The price impact of order book events. *Journal
  of Financial Econometrics*, 12(1), 47-88. doi:10.1093/jjfinec/nbt003
---

# `ContOFI`

## Description

`ContOFI` is the order-flow imbalance of Cont, Kukanov, and Stoikov (2014),
computed from top-of-book (L1) events rather than from trades. The inputs are
`bid`, `ask`, `bid_size` and `ask_size`, in that order. On each update it
measures the signed change in resting depth:

    e_bid = bid_size * 1(bid >= prev_bid) - prev_bid_size * 1(bid <= prev_bid)
    e_ask = ask_size * 1(ask <= prev_ask) - prev_ask_size * 1(ask >= prev_ask)
    OFI   = e_bid - e_ask.

A bid that ticks up contributes its full size (buyers stepping in front), a bid
that ticks down removes the vanished size, and an unchanged bid contributes the
size change; the ask side is symmetric. This is the order-book OFI, distinct
from the trade-flow [`OFI`](OFI.md). The first event has no previous quote to
compare against and returns `NaN`.

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
    from screamer import ContOFI

    rng = np.random.default_rng(0)
    n = 300
    # a mid that drifts with cumulative book pressure, quoted around it
    pressure = np.cumsum(rng.standard_normal(n))
    mid = 100 + 0.02 * pressure
    bid, ask = mid - 0.05, mid + 0.05
    bid_size = np.abs(rng.standard_normal(n)) + 1 + 0.3 * pressure.clip(0)
    ask_size = np.abs(rng.standard_normal(n)) + 1 - 0.3 * pressure.clip(None, 0)
    ofi = ContOFI()(bid, ask, bid_size, ask_size)
    cum = np.cumsum(np.nan_to_num(ofi))          # running book order-flow imbalance

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=mid, mode='lines', name='mid price',
                             line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=cum, mode='lines', name='cumulative ContOFI',
                             line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(title='ContOFI: book order flow tracks the mid',
                      yaxis=dict(title='mid'), yaxis2=dict(title='sum OFI'),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
