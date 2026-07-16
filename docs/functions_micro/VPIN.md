---
name: VPIN
title: VPIN Order-Flow Toxicity
implementation_family: micro
topics:
- microstructure
tags:
- vpin
- toxicity
- informed trading
- order flow
- volume clock
- microstructure
short: "Volume-synchronized probability of informed trading (Easley-Lopez de Prado-O'Hara 2012): order-flow toxicity."
inputs: 2
outputs: 1
parameters:
- name: bucket_volume
  type: float
  default: 1.0
  description: Volume that fills one bucket (the volume clock). Set it to your instrument's volume scale.
- name: n_buckets
  type: int
  default: 50
  description: Number of trailing buckets averaged.
nan_policy: ignore
see_also:
- OFI
- BulkVolumeClassifier
---

# `VPIN`

## Description

`VPIN` (Volume-Synchronized Probability of Informed Trading; Easley, Lopez de
Prado, O'Hara 2012) measures order-flow **toxicity**: how one-sided trading is,
measured on a volume clock rather than a wall clock. Trades are packed into
equal-volume buckets of `bucket_volume`; each closed bucket contributes its
absolute order imbalance `|buy - sell|`, and `VPIN` is the mean of that imbalance
over the last `n_buckets` buckets, normalized by the bucket volume, giving a
value in `[0, 1]`. A high value means flow is dominated by one side, a sign of
informed or toxic trading that often precedes adverse price moves.

Feed per-trade buy and sell volume (from a signing rule such as `TickRuleSign` or
`BulkVolumeClassifier`). A trade that straddles a bucket boundary is split
proportionally across buckets, so the volume clock is exact. The output is `NaN`
until `n_buckets` buckets have closed. References: Easley, Lopez de Prado, O'Hara
(2012), "Flow Toxicity and Liquidity in a High-Frequency World".

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import VPIN

    rng = np.random.default_rng(0)
    n = 600
    buy = rng.exponential(1.0, n)
    sell = rng.exponential(1.0, n)
    sell[250:350] *= 0.2                       # a one-sided (toxic) buying burst
    vpin = VPIN(bucket_volume=20.0, n_buckets=20)(buy, sell)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=buy - sell, mode='lines', name='net flow (buy - sell)',
                             line=dict(color='lightslategray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=vpin, mode='lines', name='VPIN',
                             line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(title='VPIN: toxicity spikes during a one-sided burst',
                      yaxis=dict(title='net flow'), yaxis2=dict(title='VPIN', range=[0, 1]),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
