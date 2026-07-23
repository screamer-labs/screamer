---
name: BulkVolumeClassifier
title: Bulk Volume Classifier (BVC)
implementation_family: micro
topics:
- trade-signing
tags:
- trade sign
- bulk volume
- bvc
- order flow
- toxicity
- microstructure
short: Buy-initiated share of a bar's volume estimated as the normal CDF of return / trailing-window volatility.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Window length in observations for the trailing standard deviation.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behaviour.
nan_policy: ignore
see_also:
- TickRuleSign
- LeeReadySign
---

# `BulkVolumeClassifier`

## Description

`BulkVolumeClassifier` implements the Bulk Volume Classification (BVC) model of
Easley, Lopez de Prado, and O'Hara (2012). It estimates the buy-initiated
fraction of a bar's volume without tick-level data, using only the bar's return
and its trailing volatility.

At each step the operator computes the standardized return
`z = return_ / sigma_t`, where `sigma_t` is the rolling standard deviation of
`return_` over the most recent `window_size` observations, and evaluates the
standard normal CDF `Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))`. The result is a
fraction in `[0, 1]`: values near 1 indicate a predominantly buy-driven bar,
values near 0 indicate a sell-driven bar, and 0.5 indicates a neutral bar.

The rolling standard deviation is tracked with two running sums, so each step
costs O(1). The operator is causal and honors `nan_policy: ignore`; a
zero-variance window leaves the classification undefined and returns `NaN`.

A common pipeline is:

1. Compute bar log-returns with `LogReturn`.
2. Feed the return series to `BulkVolumeClassifier` to obtain a per-bar
   buy-fraction estimate.
3. Multiply by the bar's total volume to recover an estimated buy volume.

*Parameters*:

- **`window_size`** (`int`, >= 2): number of observations in the trailing
  standard deviation window.
- **`start_policy`** (`str`, default `"strict"`): controls the warmup period
  before `window_size` observations have been seen. `"strict"` emits `NaN`.
  `"expanding"` uses all available observations. `"zero"` fills with zero.

*Return value*: the buy fraction at each time step, in `[0, 1]`. `NaN` during
warmup (under `strict`) or when the input return is `NaN`.

**Reference**: Easley, D., Lopez de Prado, M. M., & O'Hara, M. (2012).
"Bulk classification of trading activity." *Working Paper*, Cornell University.

<!-- HELP_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BulkVolumeClassifier

    rng = np.random.default_rng(3)
    n = 300
    drift = np.concatenate([np.full(n // 2, 0.004), np.full(n - n // 2, -0.004)])
    ret = drift + rng.standard_normal(n) * 0.01     # up-trend then down-trend
    price = 100 * np.exp(np.cumsum(ret))
    buy_frac = BulkVolumeClassifier(30)(ret)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode='lines', name='price',
                             line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=buy_frac, mode='lines', name='buy fraction',
                             line=dict(color='seagreen')), row=2, col=1)
    fig.add_hline(y=0.5, line=dict(color='gray', dash='dot'), row=2, col=1)
    fig.update_layout(title='BulkVolumeClassifier: estimated buy share (>0.5 buy-driven)',
                      yaxis=dict(title='price'), yaxis2=dict(title='buy fraction', range=[0, 1]),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
