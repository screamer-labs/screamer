---
name: SignedVolume
title: Signed Volume
implementation_family: micro
topics:
- trade-signing
tags:
- signed volume
- order flow
- flow
- microstructure
short: Aggressor-signed volume, sign * volume.
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `SignedVolume`

## Description

`SignedVolume` multiplies a trade sign by volume to give aggressor-signed order
flow. Pair a sign source (`TickRuleSign`, or an aggressor flag) with volume, then
feed the result to `RollingKyleLambda` for price impact.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import SignedVolume

    rng = np.random.default_rng(2)
    n = 120
    sign = rng.choice([-1.0, 1.0], size=n)
    size = rng.exponential(1.0, size=n)
    flow = SignedVolume()(sign, size)

    colors = np.where(flow >= 0, 'seagreen', 'crimson')
    fig = go.Figure(go.Bar(y=flow, marker_color=colors, name='signed volume'))
    fig.update_layout(title='SignedVolume: buyer-initiated (+) and seller-initiated (-) flow',
                      xaxis_title='trade', yaxis_title='signed volume',
                      margin=dict(l=20, r=20, t=60, b=20))
    fig.show()
```
