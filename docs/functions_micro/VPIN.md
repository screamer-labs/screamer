---
name: VPIN
title: VPIN Order-Flow Toxicity
implementation_family: micro
topics:
- order-flow-imbalance
tags:
- vpin
- toxicity
- flow toxicity
- order flow toxicity
- informed trading
- probability of informed trading
- order flow
- volume clock
- volume bars
- microstructure
short: 'Volume-synchronized probability of informed trading (Easley-López de Prado-O''Hara
  2012): order-flow toxicity.'
inputs: 2
outputs: 1
parameters:
- name: bucket_volume
  type: float
  default: 1.0
  description: Volume that fills one bucket (the volume clock). Set it to your instrument's
    volume scale.
- name: n_buckets
  type: int
  default: 50
  min: 1
  description: Number of most recent closed buckets averaged.
nan_policy: ignore
see_also:
- BulkVolumeClassifier
- OFI
- TickRuleSign
- RollingOrderImbalance
references:
- Easley, D., López de Prado, M. M. and O'Hara, M. (2012). Flow toxicity and liquidity in
  a high-frequency world. *Review of Financial Studies*, 25(5), 1457-1493. doi:10.1093/rfs/hhs053
---

# `VPIN`

## Description

`VPIN` (volume-synchronized probability of informed trading; Easley, López de
Prado, and O'Hara 2012) measures order-flow toxicity: how one-sided trading is,
measured on a volume clock instead of a wall clock. The inputs are per-trade
`buy` and `sell` volume, in that order, from a signing rule such as
`TickRuleSign` or `BulkVolumeClassifier`.

Trades are packed into buckets that each hold `bucket_volume` of volume; a trade
that straddles a bucket boundary is split proportionally across the two buckets.
Each closed bucket contributes its absolute imbalance `|buy - sell|`, and `VPIN`
is the mean of that imbalance over the last `n_buckets` buckets divided by
`bucket_volume`, a value in `[0, 1]`. A high value means flow is dominated by one
side, a sign of informed trading that often precedes adverse price moves. The
output is `NaN` until `n_buckets` buckets have closed.

A volume that is not finite (`inf`, or a buy and sell volume whose sum passes the
double range) is a missing value, as `NaN` is: the output at that step is `NaN`
and no bucket moves. A finite trade that fills more than `n_buckets` buckets
replaces the whole window at once, so every step takes the same short time
whatever its volume.

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
