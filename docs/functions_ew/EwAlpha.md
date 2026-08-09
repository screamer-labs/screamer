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
- pair
short: EW regression intercept, ewmean(x) - EwBeta(x,y) * ewmean(y).
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

# `EwAlpha`

## Description

`EwAlpha` is the exponentially weighted regression intercept, the EW analog of [`RollingAlpha`](../functions_fin/RollingAlpha.md) and the companion to [`EwBeta`](EwBeta.md):

$$
\text{EwAlpha}(x, y) = \overline{x}_{ew} - \beta_{ew}\, \overline{y}_{ew}
$$

where `beta_ew = EwBeta(x, y)` and the means are exponentially weighted. This is a **2-input, 1-output** function (`FunctorBase<_, 2, 1>`).

## Argument-order convention

The **first argument is the target**, the **second is the regressor**, matching `EwBeta`. Together, `EwAlpha` and `EwBeta` give the fitted line `x ≈ alpha + beta * y` under exponential weighting.

## Parameters

Specify exactly one of the following to set the smoothing factor `alpha`:

- **`com`**: Center of mass. `alpha = 1 / (1 + com)`
- **`span`**: Span. `alpha = 2 / (span + 1)`
- **`halflife`**: Half-life. `alpha = 1 - exp(-log(2) / halflife)`
- **`alpha`**: Directly sets the smoothing factor, `0 < alpha < 1`

The output is `NaN` while `EwBeta` is undefined (regressor with zero variance over the effective window).


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

## Reference

Composes [`EwBeta`](EwBeta.md) and two [`EwMean`](EwMean.md); `O(1)` per step. The EW
counterpart of `RollingAlpha`.
