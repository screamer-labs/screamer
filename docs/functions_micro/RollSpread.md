---
name: RollSpread
title: Roll Effective Spread
implementation_family: micro
topics:
- price-impact
tags:
- spread
- roll
- effective spread
- liquidity
- microstructure
short: "Roll (1984) effective spread from trade prices alone: 2*sqrt(-cov(dP_t, dP_{t-1})) over a trailing window."
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Window length in observations.
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
- RollingCov
- RollingSpread
---

# `RollSpread`

## Description

Roll (1984) showed that in a market with a fixed bid-ask spread, the serial
covariance of consecutive price changes is negative and equal to minus the
square of the half-spread. The effective spread can therefore be estimated
from trade prices alone, without a quote feed:

`2 * sqrt(-cov(dP_t, dP_{t-1}))`

where `dP_t = P_t - P_{t-1}` is the price change and the covariance is
computed over a trailing window.

`RollSpread(window_size, start_policy)(price)` computes this estimate at
each time step. When the serial covariance is non-negative (no bid-ask
bounce detected, or a trending period), the estimate is undefined and
the output is `NaN`.

Internally the operator:

- forms consecutive price changes `dP_t = price_t - price_{t-1}`;
- pairs each `dP_t` with its one-step lag `dP_{t-1}`;
- maintains a rolling sample covariance of that pair over the window (three
  running sums, O(1) per step).

The operator is causal and honors `nan_policy: ignore`.

*Parameters*:

- **`window_size`** (`int`, >= 2): number of observations in the trailing
  covariance window.
- **`start_policy`** (`str`, default `"strict"`): controls warmup behavior.
  `"strict"` emits `NaN` until the window is full. `"expanding"` uses however
  many observations are available. `"zero"` fills the warmup period with zero.

*Return value*: the Roll effective spread estimate at each time step. `NaN`
during warmup (under `strict`) or whenever the trailing serial covariance
is non-negative.

**Reference**: Roll, R. (1984). "A Simple Implicit Measure of the Effective
Bid-Ask Spread in an Efficient Market." *Journal of Finance*, 39(4), 1127-1139.

## Examples

### Basic usage

```python
import numpy as np
from screamer import RollSpread

# Synthetic bid-ask bounce: price alternates between bid and ask
price = 100.0 + 0.05 * np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
spread = RollSpread(window_size=8)(price)
print(spread[-1])   # approximately 0.1 (the full spread)
```

<!-- HELP_END -->

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollSpread

    rng = np.random.default_rng(9)
    n = 300
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.02)
    half_spread = 0.10
    price = mid + rng.choice([-half_spread, half_spread], size=n)   # bid-ask bounce
    spread = RollSpread(50)(price)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode='lines', name='trade price',
                             line=dict(color='lightslategray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=spread, mode='lines', name='Roll spread',
                             line=dict(color='steelblue')), row=2, col=1)
    fig.add_hline(y=2 * half_spread, line=dict(color='crimson', dash='dot'),
                  annotation_text='true spread = 0.20', row=2, col=1)
    fig.update_layout(title='RollSpread: effective spread from trade prices alone',
                      yaxis=dict(title='price'), yaxis2=dict(title='spread'),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
