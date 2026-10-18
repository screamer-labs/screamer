---
name: EwBeta
title: Exponentially-weighted beta
implementation_family: ew
topics:
- regression
tags:
- ew
- beta
- hedge ratio
- regression
- slope
- capm
- exponential
short: 'EW CAPM beta: cov(target, regressor) / var(regressor).'
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
- EwAlpha
- EwSpread
- EwCov
- EwMultiRegression
- RollingBeta
- KalmanRegression
nan_policy: ignore
---

# `EwBeta`

## Description

`EwBeta` computes the exponentially weighted regression slope (beta, or hedge ratio) of the first input `x` on the second input `y`:

$$
\text{EwBeta}(x, y) = \frac{\operatorname{Cov}_{ew}(x, y)}{\operatorname{Var}_{ew}(y)}
$$

The first argument is the target and the second is the regressor, as in [`RollingBeta`](../functions_fin/RollingBeta.md): `EwBeta(asset_returns, market_returns)` returns the asset's beta to the market. The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`.

With $\bar{x} = S_x / S_w$ and $\bar{y} = S_y / S_w$, where every running sum decays by $(1-\alpha)$ per step:

$$
\text{EwBeta}[t] = \frac{\dfrac{S_{xy}}{S_w} - \bar{x}\,\bar{y}}
                     {\dfrac{S_{yy}}{S_w} - \bar{y}^2}
$$

This equals `EwCov(x, y) / EwVar(y)`, and pandas `x.ewm(adjust=True).cov(y) / y.ewm(adjust=True).var()`: the bias-correction factors cancel, so they are left out. The output is `NaN` when the denominator is not positive, that is when `y` is constant over the effective window, which includes the first sample.

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
    from screamer import EwBeta

    rng = np.random.default_rng(0)
    N = 300
    market = np.cumsum(rng.standard_normal(N))
    asset = 1.5 * market + np.cumsum(0.3 * rng.standard_normal(N))
    ewbeta = EwBeta(span=60)(asset, market)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=asset, name='asset', line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=market, name='market', line=dict(color='orange')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=ewbeta, name='EwBeta(span=60)',
                             line=dict(color='crimson')), row=2, col=1)
    fig.update_layout(
        title='EwBeta: rolling CAPM beta of asset on market (true beta = 1.5)',
        yaxis=dict(title='price level'),
        yaxis2=dict(title='beta'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->
