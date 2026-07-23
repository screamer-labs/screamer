---
name: EwBeta
title: Exponentially-weighted beta
implementation_family: ew
topics:
- regression
tags:
- ew
- beta
- pair
- capm
short: 'EW CAPM beta: cov(target, regressor) / var(regressor).'
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
  description: Span (alpha = 2 / (span + 1)). Default smoothing parameter. Exclusive
    with com/halflife/alpha.
- name: halflife
  type: float
  default: null
  description: Halflife (alpha = 1 - 0.5^(1/halflife)). Exclusive with com/span/alpha.
- name: alpha
  type: float
  default: null
  description: Smoothing parameter directly. Exclusive with com/span/halflife.
nan_policy: ignore
---

# `EwBeta`

## Description

`EwBeta` computes the exponentially weighted regression slope of `x` on `y`:

$$
\text{EwBeta}(x, y) = \frac{\operatorname{Cov}(x, y)}{\operatorname{Var}(y)}
$$

This is a **2-input, 1-output** function (`FunctorBase<_, 2, 1>`).

## Argument-order convention

The **first argument is the dependent (target)**, the **second is the regressor**. Calling `EwBeta(asset_returns, market_returns)` returns the asset's beta with respect to the market, matching the CAPM definition and the existing `RollingBeta` convention.

This is the *opposite* of pandas (`pandas.ols`-style and most pandas docs use slope of `y` on `x`).

## Parameters

Specify exactly one of the following to set the smoothing factor `alpha`:

- **`com`**: Center of mass. `alpha = 1 / (1 + com)`
- **`span`**: Span. `alpha = 2 / (span + 1)`
- **`halflife`**: Half-life. `alpha = 1 - exp(-log(2) / halflife)`
- **`alpha`**: Directly sets the smoothing factor, `0 < alpha < 1`

The first sample is `NaN`. Subsequent NaNs occur when the regressor has zero variance over the effective window.

## Formula

The bias-correction factor that `EwVar` and `EwCov` apply cancels between numerator and denominator, so `EwBeta` uses the simpler unbiased form. With $\bar{x} = S_x / S_w$ and $\bar{y} = S_y / S_w$:

$$
\text{EwBeta} = \frac{\dfrac{S_{xy}}{S_w} - \bar{x}\,\bar{y}}
                     {\dfrac{S_{yy}}{S_w} - \bar{y}^2}
$$

If the denominator is zero or non-positive, the output is `NaN`.

## Identity check

By definition `EwBeta(x, y) == EwCov(x, y) / EwVar(y)`. Verified to ~1e-12 in the test suite.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
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

## Reference

Equivalent to `pandas.Series.ewm(adjust=True).cov(market) / market.ewm(adjust=True).var()`. Pandas does not expose an `ewm.beta()` method directly.
