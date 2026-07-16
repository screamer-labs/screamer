---
name: RollingOrderImbalance
title: Rolling Order Imbalance
implementation_family: micro
topics:
- order-flow-imbalance
tags:
- order imbalance
- order flow
- imbalance
- microstructure
short: Trailing-window sum of signed order flow (Chordia-Roll-Subrahmanyam imbalance).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
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
- RollingSum
- OFI
---

# `RollingOrderImbalance`

## Description

`RollingOrderImbalance` accumulates signed order flow over a trailing window.
A positive value means buyers dominated the window; a negative value means
sellers did. It is the Chordia-Roll-Subrahmanyam (2002) order imbalance
measure: the raw net signed flow over a look-back period.

`RollingOrderImbalance(window_size, start_policy)(signed_flow)` returns the
rolling sum of `signed_flow` over the most recent `window_size` observations.
It is a documented specialization of `RollingSum`: internally it calls
`RollingSum(window_size, start_policy)(signed_flow)`.

A common pipeline is:

1. Classify each trade with `TickRuleSign` or `LeeReadySign`.
2. Multiply by trade volume with `SignedVolume` to get signed flow per trade.
3. Aggregate over a bar and feed to `RollingOrderImbalance` for a rolling
   order-book pressure signal.

*Parameters*:

- **`window_size`** (`int`, >= 1): size of the trailing window.
- **`start_policy`** (`str`, default `"strict"`): controls warmup behavior.
  `"strict"` emits `NaN` until `window_size` observations have been seen.
  `"expanding"` uses however many observations are available. `"zero"` fills
  the warmup period with zero.

*Return value*: the rolling sum of signed flow at each time step. `NaN`
during warmup under `"strict"`.

**Reference**: Chordia, T., Roll, R., and Subrahmanyam, A. (2002). "Order
Imbalance, Liquidity, and Market Returns." *Journal of Financial Economics*,
65(1), 111-130.

<!-- HELP_END -->

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
