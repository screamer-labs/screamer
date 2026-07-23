---
name: EwKyleLambda
title: Kyle's Lambda (exponentially weighted)
implementation_family: micro
topics:
- price-impact
- regression
tags:
- price impact
- illiquidity
- market impact
- kyle
- lambda
- microstructure
- ew
short: Exponentially-weighted price-impact slope of return on signed order flow.
inputs: 2
outputs: 1
parameters:
- name: com
  type: float
  default: null
  description: Center of mass (alpha = 1 / (1 + com)). Exclusive with span/halflife/alpha.
- name: span
  type: float
  default: 20.0
  description: Span (alpha = 2 / (span + 1)). Default smoothing parameter. Exclusive with com/halflife/alpha.
- name: halflife
  type: float
  default: null
  description: Half-life in periods (alpha = 1 - 0.5^(1/halflife)). Exclusive with com/span/alpha.
- name: alpha
  type: float
  default: null
  description: Direct smoothing factor in (0, 1]. Exclusive with com/span/halflife.
nan_policy: ignore
see_also:
- EwBeta
- RollingKyleLambda
---

# `EwKyleLambda`

## Description

Kyle's lambda (Kyle 1985) is the slope of price change on signed order flow,
the price-impact or illiquidity coefficient: a high value signals an illiquid
market with large price impact per unit of net flow, a low value signals a
liquid one. `EwKyleLambda` estimates it using exponential weighting rather than
a fixed rolling window, so recent observations receive higher weight and the
estimate adapts faster to changing liquidity conditions than `RollingKyleLambda`.

`EwKyleLambda(span=...)(signed_flow, return_)` returns the EW regression slope
of `return_` on `signed_flow`. It is a documented specialization of `EwBeta`:
internally it calls `EwBeta(com=com, span=span, halflife=halflife, alpha=alpha)(return_, signed_flow)`.

The decay rate is specified by exactly one of the following mutex parameters:

- **`com`** (`float`): Center of mass; `alpha = 1 / (1 + com)`.
- **`span`** (`float`): EW span; `alpha = 2 / (span + 1)`.
- **`halflife`** (`float`): Half-life in periods; `alpha = 1 - exp(-log(2) / halflife)`.
- **`alpha`** (`float`): Direct smoothing factor in `(0, 1]`; weight of the current observation.

Passing none of these or more than one raises an error from the underlying `EwBeta` constructor.

*Return value*: the price-impact coefficient lambda at each time step. The
first output is `NaN` (EwBeta warmup). Subsequent `NaN` values occur when
`signed_flow` has zero variance over the effective EW window.

Because the implementation delegates entirely to `EwBeta`, causality and the
`nan_policy: ignore` contract are inherited from the C++ engine.

**Reference**: Kyle, A. S. (1985). "Continuous Auctions and Insider Trading."
*Econometrica*, 53(6), 1315-1335.

<!-- HELP_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import EwKyleLambda, RollingKyleLambda

    rng = np.random.default_rng(7)
    n = 400
    flow = rng.standard_normal(n)
    # the true impact slope steps up halfway through
    true_lambda = np.where(np.arange(n) < n // 2, 0.3, 0.7)
    ret = true_lambda * flow + rng.standard_normal(n) * 0.5
    ew = EwKyleLambda(span=60.0)(flow, ret)
    roll = RollingKyleLambda(60)(flow, ret)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=true_lambda, mode='lines', name='true lambda',
                             line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(y=roll, mode='lines', name='rolling',
                             line=dict(color='lightslategray')))
    fig.add_trace(go.Scatter(y=ew, mode='lines', name='exponentially weighted',
                             line=dict(color='steelblue')))
    fig.update_layout(title='EwKyleLambda: adapts faster to a change in liquidity',
                      xaxis_title='observation', yaxis_title="Kyle's lambda",
                      margin=dict(l=20, r=20, t=60, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
