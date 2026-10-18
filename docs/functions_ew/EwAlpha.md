---
name: EwAlpha
title: Exponentially-weighted alpha
implementation_family: ew
topics:
- regression
tags:
- ew
- alpha
- intercept
- regression
- capm
- exponential
short: EW regression intercept, ewmean(x) - EwBeta(x,y) * ewmean(y).
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
see_also:
- EwBeta
- EwSpread
- RollingAlpha
- EwMean
nan_policy: ignore
---

# `EwAlpha`

## Description

`EwAlpha` computes the exponentially weighted regression intercept of the first input `x` on the second input `y`. It is the exponentially weighted counterpart of [`RollingAlpha`](../functions_fin/RollingAlpha.md) and the companion of [`EwBeta`](EwBeta.md):

$$
\text{EwAlpha}(x, y) = \overline{x}_{ew} - \beta_{ew}\, \overline{y}_{ew}
$$

where $\beta_{ew}$ = `EwBeta(x, y)` and the means are exponentially weighted with the same smoothing factor. The first argument is the target and the second is the regressor, as in `EwBeta`; together they give the fitted line `x ≈ alpha + beta * y`. The smoothing factor `alpha` is set by exactly one of the `com`, `span`, `halflife` or `alpha` parameters (the `alpha` parameter is the smoothing factor, unrelated to the intercept this function returns).

The output is `NaN` while `EwBeta` is undefined, that is when `y` is constant over the effective window, which includes the first sample. It is also `NaN` once `y` has stopped varying, after roughly 45 / alpha rows, and for a while after a single extreme value of `y` (see `EwBeta`).

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import EwAlpha

    rng = np.random.default_rng(0)
    N = 300
    market = np.cumsum(rng.standard_normal(N))
    asset = 3.0 + 1.5 * market + np.cumsum(0.3 * rng.standard_normal(N))  # true alpha=3
    ewalpha = EwAlpha(span=60)(asset, market)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=asset, name='asset'), row=1, col=1)
    fig.add_trace(go.Scatter(y=market, name='market'), row=1, col=1)
    fig.add_trace(go.Scatter(y=ewalpha, name='EwAlpha(span=60)',
                             line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(
        title='EwAlpha: EW regression intercept (true alpha = 3)',
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.update_yaxes(title_text='level', row=1, col=1)
    fig.update_yaxes(title_text='alpha', row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
