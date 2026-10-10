---
name: LeeReadySign
title: Lee-Ready Trade Sign
implementation_family: micro
topics:
- trade-signing
tags:
- trade sign
- trade classification
- lee ready
- lee-ready
- quote rule
- aggressor side
- buy sell classification
- microstructure
short: 'Trade sign by the Lee-Ready (1991) rule: quote test with tick-rule fallback.'
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- TickRuleSign
- BulkVolumeClassifier
- SignedVolume
- EffectiveSpread
references:
- Lee, C. M. C. and Ready, M. J. (1991). Inferring trade direction from intraday data. *Journal
  of Finance*, 46(2), 733-746.
---

# `LeeReadySign`

## Description

`LeeReadySign` classifies each trade as buyer-initiated (`+1`) or
seller-initiated (`-1`) with the Lee-Ready (1991) algorithm. The inputs are the
trade `price` and the prevailing `mid` quote, in that order. The rule has two
steps.

1. **Quote test**: a trade above the mid is a buy (`+1`) and a trade below the
   mid is a sell (`-1`).
2. **Tick-rule fallback**: a trade exactly at the mid takes the sign of the most
   recent nonzero price change (the tick rule). An up-tick is a buy and a
   down-tick a sell; an unchanged price keeps the previous sign.

The tick-rule state advances on every price, including trades decided by the
quote test, so batch and one-sample-at-a-time calls give the same signs. An
at-mid trade with no earlier price change to fall back on, such as the first
sample, gives `NaN`.

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
