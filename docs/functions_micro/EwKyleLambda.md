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
- kyle lambda
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
  description: Center of mass c, giving alpha = 1 / (1 + c). Must be positive. Give exactly
    one of com, span, halflife and alpha.
- name: span
  type: float
  default: 20.0
  description: Span s in samples, giving alpha = 2 / (s + 1). Must be greater than 1. Give
    exactly one of com, span, halflife and alpha. The default shown is an example value; the
    constructor has no default.
- name: halflife
  type: float
  default: null
  description: Half-life h in samples, giving alpha = 1 - exp(-ln(2) / h). Must be positive.
    Give exactly one of com, span, halflife and alpha.
- name: alpha
  type: float
  default: null
  description: Smoothing factor, with 0 < alpha < 1; a larger alpha gives recent samples more
    weight. Give exactly one of com, span, halflife and alpha.
nan_policy: ignore
see_also:
- RollingKyleLambda
- EwBeta
- AmihudIlliquidity
- SignedVolume
references:
- Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315-1335.
  doi:10.2307/1913210
---

# `EwKyleLambda`

## Description

Kyle's lambda (Kyle 1985) is the slope of price change on signed order flow,
also called the price-impact or illiquidity coefficient. A high value signals an
illiquid market with large price impact per unit of net flow, a low value a
liquid one. `EwKyleLambda` estimates it with exponential weighting instead of a
fixed window, so recent observations receive more weight and the estimate adapts
to changing liquidity faster than [`RollingKyleLambda`](RollingKyleLambda.md).

`EwKyleLambda(span=...)(signed_flow, return_)` returns the exponentially
weighted regression slope of `return_` on `signed_flow`. It equals
`EwBeta(span=...)(return_, signed_flow)`, with the inputs in the opposite order.
Exactly one of `com`, `span`, `halflife` and `alpha` sets the decay; passing none
or more than one raises an error.

`EwKyleLambda` is a C++ operator that runs the `EwBeta` kernel with the two
inputs swapped. It takes every input form of a two-input operator: two separate
inputs, one packed `(T, ..., 2)` array whose last axis holds `signed_flow` then
`return_`, a list of events, or an iterator. In Python the inputs also work as
keywords, `EwKyleLambda(span=20)(signed_flow=f, return_=r)`.
JavaScript has no keyword arguments, so pass the two inputs in order.

The first output is `NaN`, since a slope needs at least two observations. Later
outputs are `NaN` when `signed_flow` has zero weighted variance, once it has
stopped varying (after roughly 45 / alpha rows), and for a while after a single
extreme value of it.

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
