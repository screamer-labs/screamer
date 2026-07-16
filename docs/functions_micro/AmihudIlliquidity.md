---
name: AmihudIlliquidity
title: Amihud Illiquidity
implementation_family: micro
topics:
- microstructure
- statistics
tags:
- illiquidity
- amihud
- liquidity
- impact
- microstructure
short: Rolling mean of |return| / notional (Amihud 2002 illiquidity ratio).
inputs: 2
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
- RollingKyleLambda
- RollingMean
---

# `AmihudIlliquidity`

## Description

The Amihud (2002) illiquidity ratio measures how much the price moves per unit
of trading volume (notional). At each bar it computes `|return| / notional`,
then takes a trailing-window mean over `window_size` observations. A large
value indicates an illiquid, high-impact regime: a small trade moves the price
substantially. A small value indicates a liquid market where large volume is
absorbed with little price effect.

`AmihudIlliquidity(window_size, start_policy)(return_, notional)` returns
`RollingMean(window_size, start_policy)(|return_| / notional)`.

The ratio is elementwise: if either `return_` or `notional` is `NaN` at a
given bar, the ratio for that bar is `NaN` and `RollingMean` skips it
(inheriting `nan_policy: ignore` from the C++ engine).

A common pipeline is:

1. Compute bar returns with `LogReturn` or `Return`.
2. Compute `notional` as price times volume for the same bar.
3. Feed both to `AmihudIlliquidity` to obtain a rolling illiquidity estimate.

*Parameters*:

- **`window_size`** (`int`, >= 2): size of the trailing window.
- **`start_policy`** (`str`, default `"strict"`): controls warmup behavior.
  `"strict"` emits `NaN` until `window_size` observations have been seen.
  `"expanding"` uses however many observations are available. `"zero"` fills
  the warmup period with zero.

*Return value*: the Amihud illiquidity estimate at each time step. `NaN`
during warmup under `"strict"`, and whenever `notional` is zero or missing.

Compared to `RollingKyleLambda` (which requires signed order-flow data),
`AmihudIlliquidity` needs only a price return and a traded volume, making it
cheap to compute from standard OHLCV bars without trade-level data.

**Reference**: Amihud, Y. (2002). "Illiquidity and stock returns: cross-section
and time-series effects." *Journal of Financial Markets*, 5(1), 31-56.

<!-- HELP_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import AmihudIlliquidity

    rng = np.random.default_rng(8)
    n = 300
    ret = rng.standard_normal(n) * 0.01
    notional = np.abs(rng.standard_normal(n)) + 1.0
    notional[130:200] *= 0.25                    # a thin, illiquid patch
    amihud = AmihudIlliquidity(30)(ret, notional)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=notional, name='notional', line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=amihud, name='Amihud illiquidity',
                             line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(title='AmihudIlliquidity: price move per dollar traded',
                      yaxis=dict(title='notional'), yaxis2=dict(title='illiquidity'),
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
