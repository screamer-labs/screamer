---
name: TickRuleSign
title: Tick Rule Sign
implementation_family: micro
topics:
- microstructure
tags:
- trade sign
- tick rule
- lee ready
- classification
- flow
- microstructure
short: Trade sign by the tick rule (+1 up-tick, -1 down-tick, carry on unchanged).
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `TickRuleSign`

## Description

`TickRuleSign` classifies each trade as buyer-initiated (`+1`) or
seller-initiated (`-1`) by the tick rule: a trade above the previous price is a
buy, below is a sell, and an unchanged price carries the previous sign. It is the
simplest trade-sign classifier and needs only the price series. References: the
tick rule, and Lee, Ready (1991), "Inferring Trade Direction from Intraday Data".

The output stays `NaN` until the first price change: the initial bar is always
`NaN` (no prior price), and if subsequent prices are all unchanged there is no
directional tick yet and no sign to carry forward.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import TickRuleSign

    rng = np.random.default_rng(0)
    price = 100 + np.cumsum(rng.standard_normal(200) * 0.05)
    sign = TickRuleSign()(price)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode='lines', name='price',
                             line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=sign, mode='lines', line_shape='hv', name='tick sign',
                             line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(title='TickRuleSign: +1 on an up-tick, -1 on a down-tick',
                      yaxis=dict(title='price'), yaxis2=dict(title='sign', dtick=1),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
