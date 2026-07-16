---
name: LeeReadySign
title: Lee-Ready Trade Sign
implementation_family: micro
topics:
- trade-signing
tags:
- trade sign
- lee ready
- classification
- flow
- microstructure
short: "Trade sign by the Lee-Ready (1991) rule: quote test with tick-rule fallback."
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- TickRuleSign
- SignedVolume
---

# `LeeReadySign`

## Description

`LeeReadySign` classifies each trade as buyer-initiated (`+1`) or
seller-initiated (`-1`) using the Lee-Ready (1991) algorithm. The rule has
two steps.

1. **Quote test**: if the trade price is above the mid-quote, it is a buy
   (`+1`); if it is below, it is a sell (`-1`).
2. **Tick-rule fallback**: if the trade price equals the mid-quote exactly,
   the sign of the most recent price change is used instead (the tick rule).
   An up-tick is a buy; a down-tick is a sell; an unchanged tick carries the
   previous sign.

`LeeReadySign()(price, mid)` returns the signed series. The tick-rule fallback
state advances on every price (not only at-mid samples), so the classification
is consistent between whole-array and one-sample-at-a-time driving
(batch == stream). A missing price or mid (`NaN`) yields `NaN`
(nan_policy: ignore).

*Return value*: an array of `+1.0` and `-1.0` trade signs (or `NaN` where
input is missing). The first sample is `NaN` when the tick-rule fallback is
needed at index 0 (no prior price to compute a tick direction).

**Reference**: Lee, C. M. C., and Ready, M. J. (1991). "Inferring Trade
Direction from Intraday Data." *Journal of Finance*, 46(2), 733-746.

<!-- HELP_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import LeeReadySign

    rng = np.random.default_rng(1)
    n = 200
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.05)
    price = mid + rng.choice([-0.5, 0.5], size=n)   # trades at the bid or the offer
    sign = LeeReadySign()(price, mid)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode='markers', name='trade price',
                             marker=dict(color='lightslategray', size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(y=mid, mode='lines', name='mid',
                             line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=sign, mode='lines', line_shape='hv', name='Lee-Ready sign',
                             line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(title='LeeReadySign: buy above the mid, sell below',
                      yaxis=dict(title='price'), yaxis2=dict(title='sign', dtick=1),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
