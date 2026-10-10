---
name: TickRuleSign
title: Tick Rule Sign
implementation_family: micro
topics:
- trade-signing
tags:
- trade sign
- trade classification
- tick rule
- tick test
- aggressor side
- buy sell classification
- microstructure
short: Trade sign by the tick rule (+1 up-tick, -1 down-tick, carry on unchanged).
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- LeeReadySign
- BulkVolumeClassifier
- SignedVolume
references:
- Lee, C. M. C. and Ready, M. J. (1991). Inferring trade direction from intraday data. *Journal
  of Finance*, 46(2), 733-746.
---

# `TickRuleSign`

## Description

`TickRuleSign` classifies each trade as buyer-initiated (`+1`) or
seller-initiated (`-1`) by the tick rule: a trade above the previous price is a
buy, a trade below it a sell, and an unchanged price keeps the previous sign. It
needs only the price series.

The output stays `NaN` until the first price change: the first sample has no
previous price, and while the price is unchanged there is no sign to carry
forward.

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
