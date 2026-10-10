---
name: AmihudIlliquidity
title: Amihud Illiquidity
implementation_family: micro
topics:
- price-impact
- statistics
tags:
- illiquidity
- amihud
- illiquidity ratio
- liquidity
- price impact
- microstructure
short: Rolling mean of |return| / notional (Amihud 2002 illiquidity ratio).
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent valid observations averaged.
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
- RollingKyleLambda
- EwKyleLambda
- RollSpread
- RollingMean
references:
- 'Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects.
  *Journal of Financial Markets*, 5(1), 31-56. doi:10.1016/S1386-4181(01)00024-6'
---

# `AmihudIlliquidity`

## Description

The Amihud (2002) illiquidity ratio measures how much the price moves per unit
of traded notional. At each step it computes `|return_| / notional` and returns
the trailing mean of that ratio over the last `window_size` observations. A
large value indicates an illiquid, high-impact market, where a small trade moves
the price substantially. A small value indicates a liquid market that absorbs
large volume with little price effect.

`AmihudIlliquidity(window_size, start_policy)(return_, notional)` equals
`RollingMean(window_size, start_policy)(abs(return_) / notional)`, except that a
step with zero `notional` gives `NaN` and is left out of the window, the same as
a `NaN` on either input.

A common pipeline is:

1. Compute bar returns with `LogReturn` or `Return`.
2. Compute `notional` as price times volume for the same bar.
3. Feed both to `AmihudIlliquidity` to obtain a rolling illiquidity estimate.

`RollingKyleLambda` needs signed order flow, while `AmihudIlliquidity` needs
only a return and a traded volume, so it can be computed from standard OHLCV
bars without trade-level data.

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
