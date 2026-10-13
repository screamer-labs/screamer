---
name: RollingOrderImbalance
title: Rolling Order Imbalance
implementation_family: micro
topics:
- order-flow-imbalance
tags:
- order imbalance
- net order flow
- signed flow
- order flow
- imbalance
- buying pressure
- microstructure
short: Trailing-window sum of signed order flow (Chordia-Roll-Subrahmanyam imbalance).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent observations summed.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- RollingSum
- OFI
- SignedVolume
- TickRuleSign
references:
- Chordia, T., Roll, R. and Subrahmanyam, A. (2002). Order imbalance, liquidity, and market
  returns. *Journal of Financial Economics*, 65(1), 111-130.
---

# `RollingOrderImbalance`

## Description

`RollingOrderImbalance` sums signed order flow over a trailing window, the order
imbalance measure of Chordia, Roll, and Subrahmanyam (2002). A positive value
means buyers dominated the window, a negative value that sellers did.

`RollingOrderImbalance(window_size, start_policy)(signed_flow)` returns the sum
of `signed_flow` over the most recent `window_size` observations. It equals
`RollingSum(window_size, start_policy)(signed_flow)`: it is a C++ operator that
runs the `RollingSum` kernel. In Python the input also works as a keyword,
`RollingOrderImbalance(20)(signed_flow=f)`.

A common pipeline is:

1. Classify each trade with `TickRuleSign` or `LeeReadySign`.
2. Multiply by trade volume with `SignedVolume` to get signed flow per trade.
3. Aggregate over a bar and feed the result to `RollingOrderImbalance` for a
   rolling buy-sell pressure signal.

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
    from screamer import RollingOrderImbalance, SignedVolume

    rng = np.random.default_rng(5)
    n = 300
    sign = rng.choice([-1.0, 1.0], size=n, p=[0.42, 0.58])   # mild buy bias
    size = rng.exponential(1.0, size=n)
    flow = SignedVolume()(sign, size)
    imbalance = RollingOrderImbalance(30)(flow)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    colors = np.where(flow >= 0, 'seagreen', 'crimson')
    fig.add_trace(go.Bar(y=flow, marker_color=colors, name='signed flow'), row=1, col=1)
    fig.add_trace(go.Scatter(y=imbalance, name='rolling imbalance',
                             line=dict(color='darkorange')), row=2, col=1)
    fig.add_hline(y=0, line=dict(color='gray', dash='dot'), row=2, col=1)
    fig.update_layout(title='RollingOrderImbalance: trailing sum of signed flow',
                      yaxis=dict(title='signed flow'), yaxis2=dict(title='imbalance'),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
