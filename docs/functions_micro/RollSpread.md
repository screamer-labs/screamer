---
name: RollSpread
title: Roll Effective Spread
implementation_family: micro
topics:
- price-impact
tags:
- roll spread
- roll measure
- bid-ask spread
- effective spread
- bid-ask bounce
- liquidity
- microstructure
short: 'Roll (1984) effective spread from trade prices alone: 2*sqrt(-cov(dP_t, dP_{t-1}))
  over a trailing window.'
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of consecutive price-change pairs in the trailing covariance window.
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
- EffectiveSpread
- RealizedSpread
- AmihudIlliquidity
- RollingCov
references:
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient
  market. *Journal of Finance*, 39(4), 1127-1139.
---

# `RollSpread`

## Description

Roll (1984) showed that in a market with a fixed bid-ask spread, the serial
covariance of consecutive price changes is negative and equal to minus the
square of the half-spread. The effective spread can therefore be estimated from
trade prices alone, without a quote feed:

    spread = 2 * sqrt(-cov(dP_t, dP_{t-1}))

where `dP_t = P_t - P_{t-1}` is the price change and the covariance of each
price change with the previous one is the sample covariance over a trailing
window of `window_size` pairs.

`RollSpread(window_size, start_policy)(price)` returns this estimate at each
step. When the serial covariance is zero or positive (no bid-ask bounce, for
example in a trending period), the estimate is undefined and the output is
`NaN`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

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
